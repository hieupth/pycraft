import json
import os
import dotenv
import numpy as np
import triton_python_backend_utils as pb_utils

from torch import topk as torch_topk
from torch.nn.functional import softmax
from vietocr.model.vocab import Vocab
from vietocr.tool.config import Cfg

dotenv.load_dotenv()
SOS_TOKEN = os.getenv('SOS_TOKEN', default=1)
EOS_TOKEN = os.getenv('EOS_TOKEN', default=2)
MAX_SEQ_LENGTH = os.getenv('MAX_SEQ_LENGTH', default=128)
CONFIG_MODEL = os.getenv('CONFIG_MODEL', 'vgg_seq2seq')
VOCAB = Vocab(Cfg.load_config_from_file(CONFIG_MODEL)['vocab'])

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        print('Initialized...')
        self.model_config = json.loads(args["model_config"])


    async def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []
        for request in requests:
            # request is only one image
            batch_images = pb_utils.get_input_tensor_by_name(request, "IMAGE")
            #TODO: image is batch
            
            cnn_requests = pb_utils.InferenceRequest(
                model_name="ocr_cnn",
                request_output_names=['OUTPUT'],
                inputs=batch_images
            )

            cnn_response = await cnn_requests.async_exec()
            src = pb_utils.get_output_tensor_by_name(cnn_response, 'OUTPUT')

            # Inference through decoder
            encoder_requests = pb_utils.InferenceRequest(
                model_name="ocr_encoder",
                request_output_names=['encoder_outputs', "hidden"],
                inputs=src
            )
            encoder_response = await encoder_requests.async_exec()
            encoder_outputs = pb_utils.get_output_tensor_by_name(encoder_response, 'encoder_outputs')
            hidden = pb_utils.get_output_tensor_by_name(encoder_response, 'hidden')

            # Inference through decoder
            number_images = len(batch_images.asnumpy())
            translated_sentence = [[SOS_TOKEN]*number_images]

            max_length = 0
            
            while max_length <= MAX_SEQ_LENGTH and not all(np.any(
                np.asarray(translated_sentence).T==EOS_TOKEN, axis=1
            )):
                tgt_inp = pb_utils.Tensor('tgt', np.asarray(translated_sentence))
                hidden = pb_utils.Tensor('hidden', hidden)
                encoder_outputs = pb_utils.Tensor('encoder_outputs', encoder_outputs)
                
                decoder_requests = pb_utils.InferenceRequest(
                    model_name="ocr_decoder",
                    request_output_names=['output', "hidden_out", "last"],
                    inputs=[tgt_inp, hidden, encoder_outputs]
                )
                decoder_response = await decoder_requests.async_exec()
                output = pb_utils.get_output_tensor_by_name(decoder_response, 'output')
                hidden = pb_utils.get_output_tensor_by_name(decoder_response, 'hidden_out')

                output = softmax(output, dim=-1)
                output = output.to('cpu')

                _, indices  = torch_topk(output, 5)
                
                indices = indices[:, -1, 0]
                indices = indices.tolist()

                translated_sentence.append(indices)   
                max_length += 1

                del output
                
            translated_sentence = np.asarray(translated_sentence).T

            sentences = self.postprocess(translated_sentence)
            output = [pb_utils.Tensor(
                "SENTENCES", 
                sentences
            )]
            
            # Output return
            response = pb_utils.InferenceResponse(
                output_tensors=output
            )
            responses.append(response)

        return responses
    

    def postprocess(self, translated_sentence):
        sentences = VOCAB.batch_decode(translated_sentence)
        return sentences