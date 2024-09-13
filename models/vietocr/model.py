import json
import os
import dotenv
import triton_python_backend_utils as pb_utils

from python.craftdet.utils import image as image_utils
from craftdet.detection import utils as craft_utils

dotenv.load_dotenv()
LONG_SIZE = int(os.getenv("LONG_SIZE"))
LOW_TEXT = float(os.getenv("LOW_TEXT"))
TEXT_THRESH = float(os.getenv("TEXT_THRESH"))
LINK_THRESH = float(os.getenv("LINK_THRESH"))
POLY = bool(os.getenv("POLY"))

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
            for 

            # Inference through CRAFTDET
            inference_request = pb_utils.InferenceRequest(
                model_name="1",
                request_output_names=['encoder_outputs', "hidden"],
                inputs=image_preprocessed
            )
            y, hidden = await inference_request.async_exec()

            # Inference through Refiner
            inference_request_refiner = pb_utils.InferenceRequest(
                model_name="refine_net",
                request_output_names=["onnx::Concat_1", "onnx::Transpose_0"],
                inputs=[y, hidden] # [ -1, 32, 512, 512 ] [ -1, 512, 512, 2 ]
            )
            y_refiner = await inference_request_refiner.async_exec()

            # Post-processing
            maps = self.postprocessing(
                y=y, 
                y_refiner=y_refiner, 
                ratio_h=ratio_h, 
                ratio_w=ratio_w, 
                images=images
            )

            # Output return
            output = []
            # output_names = ['boxes', 'boxes_as_ratio', 'polys', 'polys_as_ratio', 'text_score_heatmap', 'link_score_heatmap']
            output_names = ['boxes', 'boxes_as_ratio']
            for e in output_names:
                output.append(pb_utils.Tensor(
                    e, 
                    maps[e]
                ))
            response = pb_utils.InferenceResponse(
                output_tensors=output
            )
            responses.append(response)

        return responses
    

    def preprocessing(images):
        # assert arbitrary image
        preprocessed_images = []
        for image in images:
            # assert image shape
            assert image.shape[1] == 3, "Image must have 3 channels at index 1"
            # resize
            img_resized, target_ratio, _ = image_utils.resize_aspect_ratio(
                image, 
                LONG_SIZE
            )
            ratio_h = ratio_w = 1 / target_ratio
            preprocessed_images.append(image_utils.norm_mean_var(img_resized))
        return preprocessed_images, ratio_w, ratio_h


    def postprocessing(y, y_refiner, ratio_w, ratio_h, images) -> dict:
        score_texts = []
        score_links = []
        boxes_images = []
        polys_images = []

        for index, image in enumerate(images):
            # Post-processing
            score_text = y[index, :, :, 0].cpu().data.numpy()
            score_link = y_refiner[index, :, :, 1].cpu().data.numpy()


            boxes, polys = craft_utils.get_det_boxes(
                score_text, score_link, TEXT_THRESH, LINK_THRESH, LOW_TEXT, poly=POLY
            )
            # coordinate adjustment
            boxes = craft_utils.adjust_result_coordinates(boxes, ratio_w, ratio_h)
            #polys = [x for x in polys if x is not None]
            # polys = craft_utils.adjust_result_coordinates(polys, ratio_w, ratio_h)
            # for k in range(len(polys)):
            #     if polys[k] is None:
            #         polys[k] = boxes[k]
            # get image size
            img_height = image.shape[0]
            img_width = image.shape[1]
            # calculate box coords as ratios to image size
            boxes_as_ratio = []
            for box in boxes:
                boxes_as_ratio.append(box / [img_width, img_height])
            # boxes_as_ratio = np.array(boxes_as_ratio)
            # calculate poly coords as ratios to image size
            # polys_as_ratio = []
            # for poly in polys:
            #     polys_as_ratio.append(poly / [img_width, img_height])
            #polys_as_ratio = np.array(polys_as_ratio)
            # generate heatmap
            # text_score_heatmap = image_utils.cv2_heatmap_image(score_text)
            # link_score_heatmap = image_utils.cv2_heatmap_image(score_link)

            # final
            score_texts.append(score_text)
            score_links.append(score_link)
            boxes_images.append(boxes)
            # polys_images.append(polys)

        
        return {
            "boxes": boxes, # [ b, num_detected, 4 ]
            "boxes_as_ratio": boxes_as_ratio, # [ b, num_detected, 4 ]
            # "polys": polys, # [ b, num_detected, 8 ]
            # "polys_as_ratio": polys_as_ratio, # [ b, num_detected, 8 ]
            # "heatmaps": {
            #     "text_score_heatmap": text_score_heatmap,
            #     "link_score_heatmap": link_score_heatmap,
            # },
        }

