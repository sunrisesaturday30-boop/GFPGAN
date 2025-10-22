import cv2
import os
import torch
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from torchvision.transforms.functional import normalize
try:
    import onnxruntime as ort  # optional; used only when loading .onnx model
except Exception:  # pragma: no cover
    ort = None

from gfpgan.archs.gfpgan_bilinear_arch import GFPGANBilinear
from gfpgan.archs.gfpganv1_arch import GFPGANv1
from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GFPGANer():
    """Helper for restoration with GFPGAN.

    It will detect and crop faces, and then resize the faces to 512x512.
    GFPGAN is used to restored the resized faces.
    The background is upsampled with the bg_upsampler.
    Finally, the faces will be pasted back to the upsample background image.

    Args:
        model_path (str): The path to the GFPGAN model. It can be urls (will first download it automatically).
        upscale (float): The upscale of the final output. Default: 2.
        arch (str): The GFPGAN architecture. Option: clean | original. Default: clean.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        bg_upsampler (nn.Module): The upsampler for the background. Default: None.
    """

    def __init__(self, model_path, upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=None, device=None):
        self.upscale = upscale
        self.bg_upsampler = bg_upsampler
        self.use_ort = False
        self.ort_session = None
        self.ort_input_names = None
        self.ort_output_names = None

        # initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        # If an ONNX model is provided, initialize ONNXRuntime CUDA session and skip torch model init
        if isinstance(model_path, str) and model_path.lower().endswith('.onnx'):
            if ort is None:
                raise ImportError('onnxruntime is required to load .onnx models but is not installed.')
            # Enforce CUDA-only provider as requested
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' not in available_providers:
                raise RuntimeError('CUDAExecutionProvider is not available in onnxruntime installation.')
            sess_options = ort.SessionOptions()
            # Create session with CUDA only
            self.ort_session = ort.InferenceSession(model_path, sess_options=sess_options, providers=['CUDAExecutionProvider'])
            self.ort_input_names = [i.name for i in self.ort_session.get_inputs()]
            self.ort_output_names = [o.name for o in self.ort_session.get_outputs()]
            self.use_ort = True
        else:
            # initialize the GFP-GAN torch model
            if arch == 'clean':
                self.gfpgan = GFPGANv1Clean(
                    out_size=512,
                    num_style_feat=512,
                    channel_multiplier=channel_multiplier,
                    decoder_load_path=None,
                    fix_decoder=False,
                    num_mlp=8,
                    input_is_latent=True,
                    different_w=True,
                    narrow=1,
                    sft_half=True)
            elif arch == 'bilinear':
                self.gfpgan = GFPGANBilinear(
                    out_size=512,
                    num_style_feat=512,
                    channel_multiplier=channel_multiplier,
                    decoder_load_path=None,
                    fix_decoder=False,
                    num_mlp=8,
                    input_is_latent=True,
                    different_w=True,
                    narrow=1,
                    sft_half=True)
            elif arch == 'original':
                self.gfpgan = GFPGANv1(
                    out_size=512,
                    num_style_feat=512,
                    channel_multiplier=channel_multiplier,
                    decoder_load_path=None,
                    fix_decoder=True,
                    num_mlp=8,
                    input_is_latent=True,
                    different_w=True,
                    narrow=1,
                    sft_half=True)
            elif arch == 'RestoreFormer':
                from gfpgan.archs.restoreformer_arch import RestoreFormer
                self.gfpgan = RestoreFormer()
        # initialize face helper
        self.face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            use_parse=True,
            device=self.device,
            model_rootpath='gfpgan/weights')

        if not self.use_ort:
            if model_path.startswith('https://'):
                model_path = load_file_from_url(
                    url=model_path, model_dir=os.path.join(ROOT_DIR, 'gfpgan/weights'), progress=True, file_name=None)
            loadnet = torch.load(model_path)
            if 'params_ema' in loadnet:
                keyname = 'params_ema'
            else:
                keyname = 'params'
            self.gfpgan.load_state_dict(loadnet[keyname], strict=True)
            self.gfpgan.eval()
            self.gfpgan = self.gfpgan.to(self.device)

    @torch.no_grad()
    def enhance(self, img, has_aligned=False, only_center_face=False, paste_back=True, weight=0.5):
        self.face_helper.clean_all()

        if has_aligned:  # the inputs are already aligned
            img = cv2.resize(img, (512, 512))
            self.face_helper.cropped_faces = [img]
        else:
            self.face_helper.read_image(img)
            # get face landmarks for each face
            self.face_helper.get_face_landmarks_5(only_center_face=only_center_face, eye_dist_threshold=5)
            # eye_dist_threshold=5: skip faces whose eye distance is smaller than 5 pixels
            # TODO: even with eye_dist_threshold, it will still introduce wrong detections and restorations.
            # align and warp each face
            self.face_helper.align_warp_face()

        # face restoration
        for cropped_face in self.face_helper.cropped_faces:
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0)

            try:
                if self.use_ort:
                    # ONNXRuntime expects numpy float32 NCHW
                    input_dict = {}
                    input_tensor = cropped_face_t.numpy().astype('float32')
                    # Try to find the main input
                    if len(self.ort_input_names) == 1:
                        input_dict[self.ort_input_names[0]] = input_tensor
                    else:
                        # Heuristic: first is image, any input containing 'weight' gets the scalar
                        for name in self.ort_input_names:
                            if 'weight' in name.lower():
                                input_dict[name] = float(weight)
                            else:
                                input_dict[name] = input_tensor
                    ort_outputs = self.ort_session.run(self.ort_output_names, input_dict)
                    # Use the first output as restored face tensor in [-1, 1]
                    output_np = ort_outputs[0]
                    # convert to torch tensor for downstream tensor2img utility
                    output_tensor = torch.from_numpy(output_np)
                    restored_face = tensor2img(output_tensor.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
                else:
                    output = self.gfpgan(cropped_face_t.to(self.device), return_rgb=False, weight=weight)[0]
                    # convert to image
                    restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
            except RuntimeError as error:
                print(f'\tFailed inference for GFPGAN: {error}.')
                restored_face = cropped_face

            restored_face = restored_face.astype('uint8')
            self.face_helper.add_restored_face(restored_face)

        if not has_aligned and paste_back:
            # upsample the background
            if self.bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = self.bg_upsampler.enhance(img, outscale=self.upscale)[0]
            else:
                bg_img = None

            self.face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img)
            return self.face_helper.cropped_faces, self.face_helper.restored_faces, restored_img
        else:
            return self.face_helper.cropped_faces, self.face_helper.restored_faces, None
