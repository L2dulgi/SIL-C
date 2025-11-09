#!/usr/bin/env python3
"""
DINOv3 Embedding Extractor (Global and Dense)

This module provides a unified interface to extract both global and dense embeddings
from images using pretrained DINOv3 models.
"""

import os
import torch
from pathlib import Path
from PIL import Image
from torchvision.transforms import v2
import torchvision.transforms.functional as TF
from typing import Union, List, Tuple, Optional
import numpy as np

# NOTE test hard coded.
os.environ.setdefault('DINOV3_REPO_DIR', str(Path('~/dinotest/dinov3').expanduser()))

class DINOv3Embedder:
    """
    DINOv3 Embedding Extractor (Global and Dense)

    Loads a pretrained DINOv3 model and extracts either global or dense embeddings from images.

    Examples:
        # Global embedding
        >>> embedder = DINOv3Embedder(model_name='ViT-B/16')
        >>> embedding = embedder.forward('path/to/image.jpg')
        >>> print(embedding.shape)  # (1, 768)

        # Dense embedding
        >>> embedder = DINOv3Embedder(model_name='ViT-B/16')
        >>> features = embedder.forward('path/to/image.jpg', mode='dense')
        >>> print(features.shape)  # (768, 48, 48) - patch-level features
    """

    # Model number of layers mapping
    MODEL_NUM_LAYERS = {
        'dinov3_vits16': 12,
        'dinov3_vits16plus': 12,
        'dinov3_vitb16': 12,
        'dinov3_vitl16': 24,
        'dinov3_vith16plus': 32,
        'dinov3_vit7b16': 40,
        'dinov3_convnext_tiny': 12,
        'dinov3_convnext_small': 12,
        'dinov3_convnext_base': 12,
        'dinov3_convnext_large': 12,
    }

    AVAILABLE_MODELS = {
        'ViT-S/16': {
            'hub_name': 'dinov3_vits16',
            'weights': 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
            'embed_dim': 384,
            'dataset': 'LVD-1689M'
        },
        'ViT-S+/16': {
            'hub_name': 'dinov3_vits16plus',
            'weights': 'dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth',
            'embed_dim': 384,
            'dataset': 'LVD-1689M'
        },
        'ViT-B/16': {
            'hub_name': 'dinov3_vitb16',
            'weights': 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
            'embed_dim': 768,
            'dataset': 'LVD-1689M'
        },
        'ViT-L/16': {
            'hub_name': 'dinov3_vitl16',
            'weights': 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
            'embed_dim': 1024,
            'dataset': 'LVD-1689M'
        },
        'ViT-H+/16': {
            'hub_name': 'dinov3_vith16plus',
            'weights': 'dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth',
            'embed_dim': 1280,
            'dataset': 'LVD-1689M'
        },
        'ViT-7B/16': {
            'hub_name': 'dinov3_vit7b16',
            'weights': 'dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth',
            'embed_dim': 4096,
            'dataset': 'LVD-1689M'
        },
        'ViT-L/16-SAT': {
            'hub_name': 'dinov3_vitl16',
            'weights': 'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
            'embed_dim': 1024,
            'dataset': 'SAT-493M'
        },
        'ViT-7B/16-SAT': {
            'hub_name': 'dinov3_vit7b16',
            'weights': 'dinov3_vit7b16_pretrain_sat493m-a6675841.pth',
            'embed_dim': 4096,
            'dataset': 'SAT-493M'
        },
        'ConvNeXt-Tiny': {
            'hub_name': 'dinov3_convnext_tiny',
            'weights': 'dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth',
            'embed_dim': 768,
            'dataset': 'LVD-1689M'
        },
        'ConvNeXt-Small': {
            'hub_name': 'dinov3_convnext_small',
            'weights': 'dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth',
            'embed_dim': 768,
            'dataset': 'LVD-1689M'
        },
        'ConvNeXt-Base': {
            'hub_name': 'dinov3_convnext_base',
            'weights': 'dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth',
            'embed_dim': 1024,
            'dataset': 'LVD-1689M'
        },
        'ConvNeXt-Large': {
            'hub_name': 'dinov3_convnext_large',
            'weights': 'dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth',
            'embed_dim': 1536,
            'dataset': 'LVD-1689M'
        },
    }

    def __init__(
        self,
        model_name: str = 'ViT-B/16',
        device: str = 'auto',
        repo_dir: str = None,
        image_size: int = 224,
        patch_size: int = 16,
    ):
        """
        Initialize DINOv3 Embedder

        Args:
            model_name: Name of the model to use (default: 'ViT-B/16')
            device: Device to use ('cpu', 'cuda', or 'auto')
            repo_dir: Path to DINOv3 repository (default: from DINOV3_REPO_DIR env var or ./dinov3)
            image_size: Input image size (default: 224 for global, 768 for dense)
            patch_size: Patch size for dense mode (default: 16)
            deterministic: If True, ensures deterministic results (default: True)
        """
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model '{model_name}' not available. "
                f"Choose from: {', '.join(self.AVAILABLE_MODELS.keys())}"
            )

        self.model_name = model_name
        self.model_config = self.AVAILABLE_MODELS[model_name]
        self.image_size = image_size
        self.patch_size = patch_size

        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Set paths
        if repo_dir is None:
            # Try environment variable first, then fall back to default
            env_repo_dir = os.getenv('DINOV3_REPO_DIR')
            if env_repo_dir:
                repo_dir = Path(env_repo_dir)
            else:
                repo_dir = Path(__file__).parent / 'dinov3'
        else:
            repo_dir = Path(repo_dir)

        self.repo_dir = repo_dir
        self.weights_path = repo_dir / 'models' / self.model_config['weights']

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {self.weights_path}\n"
                f"Please download the model weights first."
            )

        # Load model
        print(f"Loading {model_name} model...")
        print(f"  Repo dir: {self.repo_dir}")
        print(f"  Weights: {self.weights_path.name}")
        print(f"  Weights path: {self.weights_path}")
        print(f"  Device: {self.device}")
        print(f"  Embedding dimension: {self.model_config['embed_dim']}")

        self.model = self._load_model()
        self.model.eval()

        # Create transform (for global mode)
        self.transform = self._create_transform()

        print("✓ Model loaded successfully!")

    def _load_model(self):
        """Load the DINOv3 model"""
        model = torch.hub.load(
            str(self.repo_dir),
            self.model_config['hub_name'],
            source='local',
            pretrained=True,
            weights=str(self.weights_path)
        )
        return model.to(self.device)

    def _create_transform(self):
        """Create image transformation pipeline for global mode"""
        # Use different normalization for satellite models
        if 'SAT' in self.model_name:
            mean = (0.430, 0.411, 0.296)
            std = (0.213, 0.156, 0.143)
        else:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)

        return v2.Compose([
            v2.ToImage(),
            v2.Resize((self.image_size, self.image_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std)
        ])

    def _create_dense_transform(self, image: Image.Image, image_size: int) -> torch.Tensor:
        """
        Create resize transform for dense mode that ensures dimensions divisible by patch size

        Args:
            image: PIL Image
            image_size: Target image size

        Returns:
            Resized tensor
        """
        w, h = image.size
        h_patches = int(image_size / self.patch_size)
        w_patches = int((w * image_size) / (h * self.patch_size))

        new_h = h_patches * self.patch_size
        new_w = w_patches * self.patch_size

        return TF.to_tensor(TF.resize(image, (new_h, new_w)))

    def _load_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
        """
        Load image from various formats

        Args:
            image: Path to image, PIL Image, or numpy array

        Returns:
            PIL Image
        """
        if isinstance(image, (str, Path)):
            return Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            return image.convert('RGB')
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert('RGB')
        else:
            raise TypeError(
                f"Unsupported image type: {type(image)}. "
                f"Expected str, Path, PIL.Image, or numpy.ndarray"
            )

    def forward(
        self,
        images: Union[str, Path, Image.Image, np.ndarray, List],
        mode: str = 'global',
        batch_size: int = 1,
        return_numpy: bool = False,
        layers: Optional[Union[int, List[int]]] = None,
        normalize_features: bool = True,
        dense_image_size: int = 768
    ) -> Union[torch.Tensor, np.ndarray, List]:
        """
        Extract embeddings from images

        Args:
            images: Single image or list of images
            mode: 'global' for global embeddings or 'dense' for patch-level features
            batch_size: Batch size for processing multiple images (global mode only)
            return_numpy: If True, return numpy array instead of torch tensor
            layers: For dense mode - which layer(s) to extract. None=last layer
            normalize_features: For dense mode - whether to L2 normalize features
            dense_image_size: Image size for dense mode (default: 768)

        Returns:
            Global mode: (num_images, embed_dim)
            Dense mode (single image): (embed_dim, num_patches_h, num_patches_w)
            Dense mode (multi-layer): List of above tensors
        """
        if mode == 'global':
            return self._forward_global(images, batch_size, return_numpy)
        elif mode == 'dense':
            # Dense mode only supports single image at a time
            if isinstance(images, list):
                raise ValueError("Dense mode currently supports single image only. Process images one at a time.")
            return self._forward_dense(images, layers, normalize_features, return_numpy, dense_image_size)
        else:
            raise ValueError(f"Invalid mode '{mode}'. Choose 'global' or 'dense'.")

    def _forward_global(
        self,
        images: Union[str, Path, Image.Image, np.ndarray, List],
        batch_size: int,
        return_numpy: bool
    ) -> Union[torch.Tensor, np.ndarray]:
        """Extract global embeddings from images"""
        # Handle single image
        if not isinstance(images, list):
            images = [images]

        embeddings = []

        # Process images in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]

            # Load and transform images
            batch_tensors = []
            for img in batch_images:
                pil_img = self._load_image(img)
                tensor = self.transform(pil_img)
                batch_tensors.append(tensor)

            # Stack into batch
            batch = torch.stack(batch_tensors).to(self.device)

            # Extract embeddings
            with torch.no_grad():
                output = self.model(batch)

            embeddings.append(output)

        # Concatenate all batches
        all_embeddings = torch.cat(embeddings, dim=0)

        if return_numpy:
            return all_embeddings.cpu().numpy()
        else:
            return all_embeddings

    def _forward_dense(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        layers: Optional[Union[int, List[int]]],
        normalize_features: bool,
        return_numpy: bool,
        dense_image_size: int
    ) -> Union[torch.Tensor, np.ndarray, List]:
        """Extract dense features from a single image"""
        # Load and preprocess image
        pil_img = self._load_image(image)
        img_tensor = self._create_dense_transform(pil_img, dense_image_size)

        # Normalize
        if 'SAT' in self.model_name:
            mean = (0.430, 0.411, 0.296)
            std = (0.213, 0.156, 0.143)
        else:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)

        img_normalized = TF.normalize(img_tensor, mean=mean, std=std)
        img_batch = img_normalized.unsqueeze(0).to(self.device)

        # Determine which layers to extract
        hub_name = self.model_config['hub_name']
        num_layers = self.MODEL_NUM_LAYERS.get(hub_name, 12)

        if layers is None:
            # Default: last layer
            layer_indices = [num_layers - 1]
            return_single = True
        elif isinstance(layers, int):
            layer_indices = [layers]
            return_single = True
        else:
            layer_indices = layers
            return_single = False

        # Validate layer indices
        for idx in layer_indices:
            if idx < 0 or idx >= num_layers:
                raise ValueError(f"Layer index {idx} out of range [0, {num_layers-1}]")

        # Extract features
        with torch.no_grad():
            feats = self.model.get_intermediate_layers(
                img_batch,
                n=layer_indices,
                reshape=True,
                norm=normalize_features
            )

        # Process features
        processed_feats = []
        for feat in feats:
            # Shape: (1, embed_dim, num_patches_h, num_patches_w)
            feat = feat.squeeze(0).detach().cpu()
            processed_feats.append(feat)

        # Return format
        if return_single:
            result = processed_feats[0]
        else:
            result = processed_feats

        if return_numpy:
            if return_single:
                return result.numpy()
            else:
                return [f.numpy() for f in result]
        else:
            return result

    def get_patch_grid_size(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        dense_image_size: int = 768
    ) -> Tuple[int, int]:
        """
        Get the patch grid size (height, width) for dense mode

        Args:
            image: Input image
            dense_image_size: Image size for dense mode

        Returns:
            Tuple of (num_patches_h, num_patches_w)
        """
        pil_img = self._load_image(image)
        w, h = pil_img.size

        h_patches = int(dense_image_size / self.patch_size)
        w_patches = int((w * dense_image_size) / (h * self.patch_size))

        return (h_patches, w_patches)

    def visualize_pca(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        n_components: int = 3,
        dense_image_size: int = 768,
        return_numpy: bool = True
    ) -> np.ndarray:
        """
        Extract dense features and apply PCA for visualization

        Args:
            image: Input image
            n_components: Number of PCA components (typically 3 for RGB)
            dense_image_size: Image size for dense mode
            return_numpy: If True, return numpy array

        Returns:
            PCA-transformed features (n_components, num_patches_h, num_patches_w)
        """
        from sklearn.decomposition import PCA

        # Extract dense features
        features = self.forward(image, mode='dense', normalize_features=True,
                              dense_image_size=dense_image_size)

        # Reshape for PCA: (embed_dim, h, w) -> (h*w, embed_dim)
        embed_dim, h, w = features.shape
        features_flat = features.view(embed_dim, -1).permute(1, 0)

        # Apply PCA
        pca = PCA(n_components=n_components, whiten=True)
        features_pca = pca.fit_transform(features_flat.numpy())

        # Reshape back: (h*w, n_components) -> (h, w, n_components)
        features_pca = features_pca.reshape(h, w, n_components)

        # Apply sigmoid for better visualization
        features_pca = torch.from_numpy(features_pca)
        features_pca = torch.nn.functional.sigmoid(features_pca.mul(2.0))

        # Permute to channel-first: (h, w, n_components) -> (n_components, h, w)
        features_pca = features_pca.permute(2, 0, 1)

        if return_numpy:
            return features_pca.numpy()
        else:
            return features_pca

    def __call__(self, *args, **kwargs):
        """Allow calling the object like a function"""
        return self.forward(*args, **kwargs)

    @property
    def embed_dim(self) -> int:
        """Get embedding dimension"""
        return self.model_config['embed_dim']

    @property
    def num_layers(self) -> int:
        """Get number of layers"""
        hub_name = self.model_config['hub_name']
        return self.MODEL_NUM_LAYERS.get(hub_name, 12)

    def __repr__(self):
        return (
            f"DINOv3Embedder(\n"
            f"  model_name='{self.model_name}',\n"
            f"  embed_dim={self.embed_dim},\n"
            f"  num_layers={self.num_layers},\n"
            f"  device='{self.device}',\n"
            f"  image_size={self.image_size},\n"
            f"  patch_size={self.patch_size}\n"
            f")"
        )


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description='DINOv3 Embedding Extractor')
    parser.add_argument('images', nargs='+', help='Path to image(s)')
    parser.add_argument('--model', default='ViT-B/16',
                       choices=list(DINOv3Embedder.AVAILABLE_MODELS.keys()),
                       help='Model to use (default: ViT-B/16)')
    parser.add_argument('--device', default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use (default: auto)')
    parser.add_argument('--mode', default='global',
                       choices=['global', 'dense'],
                       help='Embedding mode (default: global)')
    parser.add_argument('--image-size', type=int, default=224,
                       help='Input image size (default: 224)')
    parser.add_argument('--dense-image-size', type=int, default=768,
                       help='Image size for dense mode (default: 768)')
    parser.add_argument('--layers', nargs='+', type=int,
                       help='Layer indices to extract (dense mode only)')
    parser.add_argument('--visualize-pca', action='store_true',
                       help='Visualize PCA of dense features')

    args = parser.parse_args()

    # Create embedder
    embedder = DINOv3Embedder(
        model_name=args.model,
        device=args.device,
        image_size=args.image_size
    )

    if args.visualize_pca:
        import matplotlib.pyplot as plt

        print("\nExtracting dense features and computing PCA...")
        pca_features = embedder.visualize_pca(
            args.images[0],
            dense_image_size=args.dense_image_size
        )

        print(f"PCA features shape: {pca_features.shape}")

        # Visualize
        plt.figure(figsize=(10, 10))
        plt.imshow(pca_features.transpose(1, 2, 0))
        plt.title(f'PCA Visualization - {args.model}')
        plt.axis('off')
        plt.show()

    elif args.mode == 'dense':
        print(f"\nExtracting dense features from {args.images[0]}...")
        features = embedder(
            args.images[0],
            mode='dense',
            layers=args.layers,
            dense_image_size=args.dense_image_size,
            return_numpy=True
        )

        if isinstance(features, list):
            print(f"\nExtracted features from {len(features)} layers:")
            for i, feat in enumerate(features):
                print(f"  Layer {args.layers[i]}: {feat.shape}")
        else:
            print(f"\nDense features shape: {features.shape}")
            print(f"  Embedding dim: {features.shape[0]}")
            print(f"  Patch grid: {features.shape[1]}x{features.shape[2]}")

    else:  # global mode
        print(f"\nExtracting global embeddings from {len(args.images)} image(s)...")

        # Extract embeddings
        embeddings = embedder(args.images, mode='global', return_numpy=True)

        print(f"\n{'='*60}")
        print("Results:")
        print(f"{'='*60}")
        print(f"Number of images: {len(args.images)}")
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Embedding dimension: {embedder.embed_dim}")
        print(f"{'='*60}")

        # Print first few values of each embedding
        for i, (img_path, emb) in enumerate(zip(args.images, embeddings)):
            print(f"\nImage {i+1}: {img_path}")
            print(f"  Embedding preview: [{emb[:5].tolist()}... (showing first 5 values)]")
            print(f"  L2 norm: {np.linalg.norm(emb):.4f}")


if __name__ == '__main__':
    main()
