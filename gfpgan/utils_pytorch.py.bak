import cv2
import numpy as np
import os
import torch
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from torchvision.transforms.functional import normalize

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

        # initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        # initialize the GFP-GAN
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
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            try:
                output = self.gfpgan(cropped_face_t, return_rgb=False, weight=weight)[0]
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

    @torch.no_grad()
    def enhance_batch(self, imgs, has_aligned=False, only_center_face=True, paste_back=True, weight=0.5, batch_size=None):
        """Batch processing for multiple images with PyTorch model (one face per image).

        Args:
            imgs (list): List of input images (numpy arrays in BGR format).
            has_aligned (bool): Whether the inputs are already aligned faces. Default: False.
            only_center_face (bool): Only restore the center face. Default: True.
            paste_back (bool): Paste the restored face back to the original image. Default: True.
            weight (float): Adjustable weight for restoration strength. Default: 0.5.
            batch_size (int, optional): Number of faces to process in each inference batch.
                If None, processes all faces in a single batch. Default: None.

        Returns:
            tuple: (list_of_cropped_faces, list_of_restored_faces, list_of_restored_images)
                Each list element corresponds to one input image.
        """
        # =====================================================================
        # FIRST PASS: Detect and align faces for all images, collect all faces
        # =====================================================================
        all_cropped_faces = []  # List of cropped faces (one per image)
        all_face_helpers = []   # Store face helper state for each image
        all_original_imgs = []  # Store original images for paste back

        for img in imgs:
            self.face_helper.clean_all()

            if has_aligned:
                # Input is already aligned, just resize to 512x512
                resized_img = cv2.resize(img, (512, 512))
                self.face_helper.cropped_faces = [resized_img]
            else:
                self.face_helper.read_image(img)
                # Get face landmarks (only center face since we have one face per image)
                self.face_helper.get_face_landmarks_5(only_center_face=only_center_face, eye_dist_threshold=5)
                # Align and warp face
                self.face_helper.align_warp_face()

            # Store the cropped face (assuming one face per image)
            if len(self.face_helper.cropped_faces) > 0:
                all_cropped_faces.append(self.face_helper.cropped_faces[0])
                # Store face helper state for paste back
                all_face_helpers.append({
                    'affine_matrices': self.face_helper.affine_matrices.copy() if hasattr(self.face_helper, 'affine_matrices') else [],
                    'inverse_affine_matrices': self.face_helper.inverse_affine_matrices.copy() if hasattr(self.face_helper, 'inverse_affine_matrices') else [],
                    'det_faces': self.face_helper.det_faces.copy() if hasattr(self.face_helper, 'det_faces') else [],
                    'input_img': self.face_helper.input_img.copy() if self.face_helper.input_img is not None else None,
                })
            else:
                # No face detected, use placeholder
                all_cropped_faces.append(None)
                all_face_helpers.append(None)

            all_original_imgs.append(img)

        # =====================================================================
        # SECOND PASS: Batch all faces and run PyTorch inference (with optional chunking)
        # =====================================================================
        # Prepare batch tensor for all valid faces
        valid_indices = [i for i, face in enumerate(all_cropped_faces) if face is not None]
        valid_faces = [all_cropped_faces[i] for i in valid_indices]

        if len(valid_faces) == 0:
            # No faces detected in any image
            return [[] for _ in imgs], [[] for _ in imgs], [None for _ in imgs]

        # Convert all faces to tensors
        face_tensors = []
        for cropped_face in valid_faces:
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            face_tensors.append(cropped_face_t)

        # Process in batches if batch_size is specified
        all_outputs = []
        if batch_size is None or batch_size <= 0:
            # Process all faces in a single batch
            batch_tensor = torch.stack(face_tensors, dim=0).to(self.device)

            try:
                output = self.gfpgan(batch_tensor, return_rgb=False, weight=weight)[0]
                # Output shape: (batch_size, 3, 512, 512)
                all_outputs.append(output)
            except RuntimeError as error:
                print(f'\tFailed batch inference for GFPGAN: {error}.')
                # Fallback: return original faces (convert back to tensor format)
                all_outputs.append(batch_tensor)
        else:
            # Process faces in chunks
            num_faces = len(face_tensors)
            for start_idx in range(0, num_faces, batch_size):
                end_idx = min(start_idx + batch_size, num_faces)
                chunk_tensors = face_tensors[start_idx:end_idx]

                # Stack into batch: (chunk_size, 3, 512, 512)
                batch_tensor = torch.stack(chunk_tensors, dim=0).to(self.device)

                try:
                    output = self.gfpgan(batch_tensor, return_rgb=False, weight=weight)[0]
                    # Output shape: (chunk_size, 3, 512, 512)
                    all_outputs.append(output)
                except RuntimeError as error:
                    print(f'\tFailed batch inference for GFPGAN (chunk {start_idx//batch_size + 1}): {error}.')
                    # Fallback: return original faces
                    all_outputs.append(batch_tensor)

        # Concatenate all batch outputs into a single tensor
        output_batch_tensor = torch.cat(all_outputs, dim=0)

        # =====================================================================
        # THIRD PASS: Split results and paste restored faces back to original images
        # =====================================================================
        all_restored_faces = [None] * len(imgs)
        all_restored_imgs = [None] * len(imgs)
        all_cropped_faces_out = [[] for _ in imgs]
        all_restored_faces_out = [[] for _ in imgs]

        for batch_idx, img_idx in enumerate(valid_indices):
            # Extract restored face from batch output
            output_tensor = output_batch_tensor[batch_idx]
            restored_face = tensor2img(output_tensor, rgb2bgr=True, min_max=(-1, 1))
            restored_face = restored_face.astype('uint8')

            all_cropped_faces_out[img_idx] = [all_cropped_faces[img_idx]]
            all_restored_faces_out[img_idx] = [restored_face]

            if not has_aligned and paste_back:
                # Restore face helper state for this image
                helper_state = all_face_helpers[img_idx]
                if helper_state is not None:
                    self.face_helper.clean_all()
                    self.face_helper.affine_matrices = helper_state['affine_matrices']
                    self.face_helper.inverse_affine_matrices = helper_state['inverse_affine_matrices']
                    self.face_helper.det_faces = helper_state['det_faces']
                    self.face_helper.input_img = helper_state['input_img']
                    self.face_helper.cropped_faces = [all_cropped_faces[img_idx]]
                    self.face_helper.restored_faces = [restored_face]

                    # Upsample background if needed
                    if self.bg_upsampler is not None:
                        bg_img = self.bg_upsampler.enhance(all_original_imgs[img_idx], outscale=self.upscale)[0]
                    else:
                        bg_img = None

                    self.face_helper.get_inverse_affine(None)
                    restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img)
                    all_restored_imgs[img_idx] = restored_img
            else:
                all_restored_imgs[img_idx] = None

        return all_cropped_faces_out, all_restored_faces_out, all_restored_imgs
