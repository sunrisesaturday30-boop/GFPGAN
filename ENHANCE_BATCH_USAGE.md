# Using `enhance_batch()` for Batch Processing

The `enhance_batch()` method allows you to process multiple images efficiently in batches using ONNX models. This is particularly useful for processing video frames (e.g., from wav2lip) where you need to restore faces in multiple images.

## Requirements

- **ONNX Model Only**: `enhance_batch()` only works with ONNX models. If you're using PyTorch models, use the regular `enhance()` method instead.
- **One Face Per Image**: The method is optimized for images containing exactly one face each.

## Basic Usage

### Initialization

```python
from gfpgan import GFPGANer
import cv2

# Initialize with ONNX model
gfpganer = GFPGANer(
    model_path='path/to/model.onnx',  # Must be .onnx file
    upscale=2,
    arch='clean'
)
```

### Simple Example

```python
# Load multiple images
images = [
    cv2.imread('frame1.jpg'),
    cv2.imread('frame2.jpg'),
    cv2.imread('frame3.jpg'),
]

# Process all images at once
cropped_faces, restored_faces, restored_imgs = gfpganer.enhance_batch(
    imgs=images,
    has_aligned=False,
    only_center_face=True,
    paste_back=True,
    weight=0.5
)

# Save restored images
for i, restored_img in enumerate(restored_imgs):
    if restored_img is not None:
        cv2.imwrite(f'output_frame_{i}.jpg', restored_img)
```

## Parameters

### Required Parameters

- **`imgs`** (list): List of input images as numpy arrays in BGR format (OpenCV format).

### Optional Parameters

- **`has_aligned`** (bool, default: `False`):
  - `False`: Input images are full images with faces that need detection and alignment.
  - `True`: Input images are already cropped and aligned faces (512x512).

- **`only_center_face`** (bool, default: `True`):
  - `True`: Only process the center face in each image (recommended for one-face-per-image scenarios).
  - `False`: Process all detected faces (not recommended for batch processing).

- **`paste_back`** (bool, default: `True`):
  - `True`: Paste the restored face back into the original image.
  - `False`: Return only the restored face without pasting back.

- **`weight`** (float, default: `0.5`):
  - Restoration strength (0.0 to 1.0).
  - Lower values (0.0-0.5): More conservative restoration, preserves original details.
  - Higher values (0.5-1.0): More aggressive restoration, better quality but may alter identity.

- **`batch_size`** (int, optional, default: `None`):
  - Number of faces to process in each inference batch.
  - `None`: Process all faces in a single batch (fastest if memory allows).
  - Integer (e.g., `8`, `16`): Process faces in chunks (better for memory-constrained scenarios).

## Return Values

The method returns a tuple of three lists:

```python
(cropped_faces, restored_faces, restored_imgs)
```

- **`cropped_faces`**: List of lists, where each inner list contains the cropped face(s) for each input image.
- **`restored_faces`**: List of lists, where each inner list contains the restored face(s) for each input image.
- **`restored_imgs`**: List of full restored images with faces pasted back (or `None` if `paste_back=False` or no face detected).

Each list element corresponds to one input image in the same order.

## Batch Size Control

### Processing All at Once (Default)

```python
# Process all images in a single batch (fastest, requires more memory)
cropped_faces, restored_faces, restored_imgs = gfpganer.enhance_batch(
    imgs=images,
    batch_size=None  # or omit this parameter
)
```

### Processing in Chunks

```python
# Process 8 faces at a time (good balance between speed and memory)
cropped_faces, restored_faces, restored_imgs = gfpganer.enhance_batch(
    imgs=images,
    batch_size=8
)

# Process 16 faces at a time (for GPUs with more memory)
cropped_faces, restored_faces, restored_imgs = gfpganer.enhance_batch(
    imgs=images,
    batch_size=16
)

# Process one at a time (slowest but uses minimal memory)
cropped_faces, restored_faces, restored_imgs = gfpganer.enhance_batch(
    imgs=images,
    batch_size=1
)
```

## Use Cases

### Processing Wav2Lip Video Frames

```python
import cv2
import os

# Load all frames from wav2lip output
frame_dir = 'wav2lip_output_frames'
frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
images = [cv2.imread(os.path.join(frame_dir, f)) for f in frame_files]

# Process all frames in batches
cropped_faces, restored_faces, restored_imgs = gfpganer.enhance_batch(
    imgs=images,
    has_aligned=False,      # Full images with faces
    only_center_face=True,   # One face per frame
    paste_back=True,         # Paste restored face back
    weight=0.5,              # Moderate restoration strength
    batch_size=8             # Process 8 frames at a time
)

# Save restored frames
output_dir = 'restored_frames'
os.makedirs(output_dir, exist_ok=True)
for i, restored_img in enumerate(restored_imgs):
    if restored_img is not None:
        cv2.imwrite(os.path.join(output_dir, f'frame_{i:04d}.jpg'), restored_img)
```

### Processing Pre-aligned Faces

```python
# If you already have cropped and aligned faces
aligned_faces = [
    cv2.imread('face1_aligned.jpg'),
    cv2.imread('face2_aligned.jpg'),
]

cropped_faces, restored_faces, restored_imgs = gfpganer.enhance_batch(
    imgs=aligned_faces,
    has_aligned=True,        # Inputs are already aligned
    paste_back=False,        # Don't paste back (already cropped)
    batch_size=16
)
```

## Error Handling

### No Face Detected

If no face is detected in an image, the corresponding output will be:
- `cropped_faces[i]`: Empty list `[]`
- `restored_faces[i]`: Empty list `[]`
- `restored_imgs[i]`: `None`

Always check for `None` before using restored images:

```python
for i, restored_img in enumerate(restored_imgs):
    if restored_img is not None:
        # Process restored image
        cv2.imwrite(f'output_{i}.jpg', restored_img)
    else:
        print(f'No face detected in image {i}')
```

### ONNX Model Required

If you try to use `enhance_batch()` with a PyTorch model, you'll get an error:

```python
# This will raise RuntimeError
gfpganer = GFPGANer(model_path='model.pth')  # PyTorch model
gfpganer.enhance_batch(imgs=images)  # RuntimeError: enhance_batch() only supports ONNX models
```

Use `enhance()` for PyTorch models instead.

## Performance Tips

1. **Batch Size Selection**:
   - Start with `batch_size=None` (process all at once) for maximum speed.
   - If you encounter out-of-memory errors, reduce `batch_size` (e.g., 8, 4, or 2).
   - Larger batch sizes are faster but require more GPU memory.

2. **Memory Management**:
   - For very long video sequences, process frames in chunks using `batch_size`.
   - Consider processing video segments separately if memory is limited.

3. **Weight Parameter**:
   - For video frames, use moderate weights (0.4-0.6) to maintain consistency across frames.
   - Higher weights may cause flickering between frames.

## Comparison with `enhance()`

| Feature | `enhance()` | `enhance_batch()` |
|---------|-------------|-------------------|
| Input | Single image | List of images |
| Model Support | PyTorch & ONNX | ONNX only |
| Processing | Sequential | Batched |
| Speed | Slower for multiple images | Faster for multiple images |
| Memory | Lower | Higher (configurable with batch_size) |

For processing multiple images, `enhance_batch()` is significantly faster than calling `enhance()` in a loop.

