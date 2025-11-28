# Using `enhance_batch()` with PyTorch Models

The `enhance_batch()` method in `utils_pytorch.py` allows you to process multiple images efficiently in batches using PyTorch models (`.pth` files).

## Requirements

- **PyTorch Model Only**: Works with PyTorch `.pth` model files.
- **One Face Per Image**: Optimized for images containing exactly one face each.

## Basic Usage

### Initialization

```python
from gfpgan.utils_pytorch import GFPGANer
import cv2

# Initialize with PyTorch model
gfpganer = GFPGANer(
    model_path='path/to/model.pth',  # PyTorch .pth file
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

- **`imgs`** (list): List of input images as numpy arrays in BGR format (OpenCV format).

- **`has_aligned`** (bool, default: `False`):
  - `False`: Input images are full images with faces that need detection and alignment.
  - `True`: Input images are already cropped and aligned faces (512x512).

- **`only_center_face`** (bool, default: `True`):
  - `True`: Only process the center face in each image (recommended for one-face-per-image scenarios).

- **`paste_back`** (bool, default: `True`):
  - `True`: Paste the restored face back into the original image.
  - `False`: Return only the restored face without pasting back.

- **`weight`** (float, default: `0.5`):
  - Restoration strength (0.0 to 1.0).
  - Lower values: More conservative restoration.
  - Higher values: More aggressive restoration.

- **`batch_size`** (int, optional, default: `None`):
  - Number of faces to process in each inference batch.
  - `None`: Process all faces in a single batch (fastest if memory allows).
  - Integer (e.g., `8`, `16`): Process faces in chunks (better for memory-constrained scenarios).

## Return Values

Returns a tuple of three lists:

```python
(cropped_faces, restored_faces, restored_imgs)
```

- **`cropped_faces`**: List of lists containing cropped face(s) for each input image.
- **`restored_faces`**: List of lists containing restored face(s) for each input image.
- **`restored_imgs`**: List of full restored images with faces pasted back (or `None` if `paste_back=False` or no face detected).

## Batch Size Examples

### Process All at Once (Default)

```python
# Process all images in a single batch (fastest, requires more memory)
cropped_faces, restored_faces, restored_imgs = gfpganer.enhance_batch(
    imgs=images,
    batch_size=None  # or omit this parameter
)
```

### Process in Chunks

```python
# Process 8 faces at a time
cropped_faces, restored_faces, restored_imgs = gfpganer.enhance_batch(
    imgs=images,
    batch_size=8
)

# Process 16 faces at a time (for GPUs with more memory)
cropped_faces, restored_faces, restored_imgs = gfpganer.enhance_batch(
    imgs=images,
    batch_size=16
)
```

## Use Case: Processing Video Frames

```python
import cv2
import os

# Load all frames
frame_dir = 'video_frames'
frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
images = [cv2.imread(os.path.join(frame_dir, f)) for f in frame_files]

# Process all frames in batches
cropped_faces, restored_faces, restored_imgs = gfpganer.enhance_batch(
    imgs=images,
    has_aligned=False,
    only_center_face=True,
    paste_back=True,
    weight=0.5,
    batch_size=8  # Process 8 frames at a time
)

# Save restored frames
output_dir = 'restored_frames'
os.makedirs(output_dir, exist_ok=True)
for i, restored_img in enumerate(restored_imgs):
    if restored_img is not None:
        cv2.imwrite(os.path.join(output_dir, f'frame_{i:04d}.jpg'), restored_img)
```

## Error Handling

If no face is detected in an image, the corresponding output will be:
- `cropped_faces[i]`: Empty list `[]`
- `restored_faces[i]`: Empty list `[]`
- `restored_imgs[i]`: `None`

Always check for `None` before using restored images:

```python
for i, restored_img in enumerate(restored_imgs):
    if restored_img is not None:
        cv2.imwrite(f'output_{i}.jpg', restored_img)
    else:
        print(f'No face detected in image {i}')
```

## Performance Tips

1. **Batch Size Selection**:
   - Start with `batch_size=None` for maximum speed.
   - If you encounter out-of-memory errors, reduce `batch_size` (e.g., 8, 4, or 2).

2. **Memory Management**:
   - Larger batch sizes are faster but require more GPU memory.
   - For very long sequences, process frames in chunks using `batch_size`.

3. **Weight Parameter**:
   - For video frames, use moderate weights (0.4-0.6) to maintain consistency across frames.

## Comparison: PyTorch vs ONNX

| Feature | `utils_pytorch.py` | `utils.py` (ONNX) |
|---------|-------------------|-------------------|
| Model Format | `.pth` files | `.onnx` files |
| Batch Processing | ✅ Supported | ✅ Supported |
| Speed | Good | Excellent (often faster) |
| Memory Usage | Moderate | Lower |

For PyTorch models, use `utils_pytorch.py`. For ONNX models, use `utils.py`.


