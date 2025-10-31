"""
Content-Based Image Retrieval (CBIR) System
============================================
A modular CBIR implementation using color moments, HSV histograms, and texture features.

Author: Optimized from notebook
Date: October 31, 2025
"""

import os
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops


class DescriptorType(Enum):
    """Available descriptor types for feature extraction."""
    COLOR_ONLY = 6
    COLOR_HISTOGRAM = 30
    FULL = 34


@dataclass
class SearchResult:
    """Container for search results."""
    index: int
    distance: float
    image: Optional[np.ndarray] = None


class FeatureExtractor:
    """Handles extraction of various image features."""
    
    def __init__(self, bins: int = 8, glcm_distances: List[int] = None, 
                 glcm_angles: List[float] = None, glcm_levels: int = 256):
        """
        Initialize feature extractor.
        
        Args:
            bins: Number of bins for histogram quantization
            glcm_distances: Distances for GLCM computation
            glcm_angles: Angles for GLCM computation
            glcm_levels: Number of gray levels for GLCM
        """
        self.bins = bins
        self.glcm_distances = glcm_distances or [1]
        self.glcm_angles = glcm_angles or [0]
        self.glcm_levels = glcm_levels
    
    def extract_color_moments(self, img: np.ndarray) -> np.ndarray:
        """
        Extract color moments (mean and std) for each channel.
        
        Args:
            img: Input image (BGR format)
            
        Returns:
            6-dimensional feature vector
        """
        img_normalized = img.astype("float32") / 255.0
        means = np.mean(img_normalized, axis=(0, 1))
        stds = np.std(img_normalized, axis=(0, 1))
        return np.concatenate([means, stds])
    
    def extract_hsv_histogram(self, img: np.ndarray) -> np.ndarray:
        """
        Extract normalized HSV color histogram.
        
        Args:
            img: Input image (BGR format)
            
        Returns:
            Histogram feature vector (bins * 3 dimensions)
        """
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_img)
        
        h_hist = np.histogram(h, bins=self.bins, range=(0, 180))[0]
        s_hist = np.histogram(s, bins=self.bins, range=(0, 256))[0]
        v_hist = np.histogram(v, bins=self.bins, range=(0, 256))[0]
        
        # Normalize histograms
        h_hist = h_hist / (h_hist.sum() + 1e-7)
        s_hist = s_hist / (s_hist.sum() + 1e-7)
        v_hist = v_hist / (v_hist.sum() + 1e-7)
        
        return np.concatenate([h_hist, s_hist, v_hist])
    
    def extract_texture_features(self, img: np.ndarray) -> np.ndarray:
        """
        Extract texture features using GLCM.
        
        Args:
            img: Input image (BGR format)
            
        Returns:
            4-dimensional texture feature vector
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.uint8)
        
        glcm = graycomatrix(
            gray, 
            distances=self.glcm_distances,
            angles=self.glcm_angles,
            levels=self.glcm_levels,
            symmetric=True,
            normed=True
        )
        
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        
        # Normalize features to [0, 1] range
        contrast = contrast / ((self.glcm_levels - 1) ** 2)
        correlation = (correlation + 1) / 2
        
        return np.array([contrast, correlation, energy, homogeneity])
    
    def extract_features(self, img: np.ndarray, 
                        descriptor_type: DescriptorType) -> np.ndarray:
        """
        Extract features based on descriptor type.
        
        Args:
            img: Input image (BGR format)
            descriptor_type: Type of descriptor to extract
            
        Returns:
            Feature vector of appropriate size
        """
        color_desc = self.extract_color_moments(img)
        
        if descriptor_type == DescriptorType.COLOR_ONLY:
            return color_desc
        
        hist_desc = self.extract_hsv_histogram(img)
        
        if descriptor_type == DescriptorType.COLOR_HISTOGRAM:
            return np.concatenate([color_desc, hist_desc])
        
        # Full descriptor
        texture_desc = self.extract_texture_features(img)
        return np.concatenate([color_desc, hist_desc, texture_desc])


class ImageTransformer:
    """Handles geometric transformations for robustness testing."""
    
    @staticmethod
    def rotate(image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))
    
    @staticmethod
    def scale(image: np.ndarray, scale_factor: float) -> np.ndarray:
        """Scale image by given factor."""
        h, w = image.shape[:2]
        new_dimensions = (int(w * scale_factor), int(h * scale_factor))
        return cv2.resize(image, new_dimensions, interpolation=cv2.INTER_LINEAR)
    
    @staticmethod
    def translate(image: np.ndarray, tx: int, ty: int) -> np.ndarray:
        """Translate image by (tx, ty) pixels."""
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        h, w = image.shape[:2]
        return cv2.warpAffine(image, M, (w, h))
    
    @staticmethod
    def flip(image: np.ndarray, flip_code: int) -> np.ndarray:
        """
        Flip image.
        
        Args:
            flip_code: 0 for vertical, 1 for horizontal, -1 for both
        """
        return cv2.flip(image, flip_code)
    
    @staticmethod
    def add_noise(image: np.ndarray, mean: float = 0, var: float = 0.01) -> np.ndarray:
        """Add Gaussian noise to image."""
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, image.shape)
        noisy = image.astype('float32') + gauss
        return np.clip(noisy, 0, 255).astype('uint8')
    
    @staticmethod
    def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image brightness by multiplication factor."""
        img = image.astype(np.float32) * factor
        return np.clip(img, 0, 255).astype(np.uint8)


class CBIRSystem:
    """Complete Content-Based Image Retrieval System."""
    
    def __init__(self, descriptor_type: DescriptorType = DescriptorType.FULL,
                 bins: int = 8):
        """
        Initialize CBIR system.
        
        Args:
            descriptor_type: Type of descriptor to use
            bins: Number of bins for histogram features
        """
        self.descriptor_type = descriptor_type
        self.feature_extractor = FeatureExtractor(bins=bins)
        self.index_matrix: Optional[np.ndarray] = None
        self.image_paths: List[str] = []
        self.images: List[np.ndarray] = []
    
    def load_images(self, folder_path: str) -> List[np.ndarray]:
        """
        Load all images from a folder.
        
        Args:
            folder_path: Path to image folder
            
        Returns:
            List of loaded images
            
        Raises:
            FileNotFoundError: If folder doesn't exist
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        images = []
        image_paths = []
        
        for filename in sorted(os.listdir(folder_path)):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
                
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            
            if img is not None:
                images.append(img)
                image_paths.append(img_path)
        
        self.image_paths = image_paths
        return images
    
    def index_database(self, dataset_path: str, verbose: bool = True) -> np.ndarray:
        """
        Index all images in the database.
        
        Args:
            dataset_path: Path to image database
            verbose: Whether to print progress information
            
        Returns:
            Feature matrix (n_images x n_features)
        """
        self.images = self.load_images(dataset_path)
        
        if verbose:
            print(f"Indexing {len(self.images)} images...")
            print(f"Descriptor type: {self.descriptor_type.name}")
            print(f"Feature vector size: {self.descriptor_type.value}")
        
        feature_list = []
        for img in self.images:
            features = self.feature_extractor.extract_features(
                img, self.descriptor_type
            )
            feature_list.append(features)
        
        self.index_matrix = np.vstack(feature_list)
        
        if verbose:
            print(f"Index matrix shape: {self.index_matrix.shape}")
        
        return self.index_matrix
    
    def search(self, query_image: np.ndarray, top_n: int = 5) -> List[SearchResult]:
        """
        Search for similar images.
        
        Args:
            query_image: Query image (BGR format)
            top_n: Number of top results to return
            
        Returns:
            List of SearchResult objects
            
        Raises:
            RuntimeError: If database hasn't been indexed
        """
        if self.index_matrix is None:
            raise RuntimeError("Database not indexed. Call index_database() first.")
        
        query_features = self.feature_extractor.extract_features(
            query_image, self.descriptor_type
        )
        
        # Compute Euclidean distances
        distances = np.linalg.norm(self.index_matrix - query_features, axis=1)
        
        # Get top N results
        top_indices = np.argsort(distances)[:top_n]
        
        results = []
        for idx in top_indices:
            results.append(SearchResult(
                index=int(idx),
                distance=float(distances[idx]),
                image=self.images[idx] if idx < len(self.images) else None
            ))
        
        return results
    
    def calculate_euclidean_distance_legacy(self, query_image: np.ndarray) -> Dict[int, float]:
        """
        Legacy method: Calculate distances using flattened image vectors.
        
        Args:
            query_image: Query image
            
        Returns:
            Dictionary mapping image index to distance
        """
        query_vector = query_image.flatten()
        query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-7)
        
        distances = {}
        for i, db_img in enumerate(self.images):
            db_vector = db_img.flatten()
            db_vector = db_vector / (np.linalg.norm(db_vector) + 1e-7)
            distance = np.linalg.norm(query_vector - db_vector)
            distances[i] = distance
        
        return dict(sorted(distances.items(), key=lambda item: item[1]))


class CBIRVisualizer:
    """Handles visualization of CBIR results."""
    
    @staticmethod
    def display_results(query_image: np.ndarray, 
                       results: List[SearchResult],
                       wait_key: bool = True) -> None:
        """
        Display query image and top results.
        
        Args:
            query_image: Query image
            results: List of search results
            wait_key: Whether to wait for key press
        """
        cv2.imshow("Query Image", query_image)
        if wait_key:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        for rank, result in enumerate(results, 1):
            print(f"Match {rank} — Index: {result.index}, "
                  f"Distance: {result.distance:.4f}")
            if result.image is not None:
                cv2.imshow(f"Match {rank}", result.image)
        
        if wait_key:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    @staticmethod
    def save_results(query_image: np.ndarray,
                    results: List[SearchResult],
                    output_path: str) -> None:
        """
        Save query and results as a grid image.
        
        Args:
            query_image: Query image
            results: List of search results
            output_path: Path to save output image
        """
        # Create a grid: query on top, results below
        n_results = len(results)
        
        # Resize all images to same size
        target_size = (200, 200)
        query_resized = cv2.resize(query_image, target_size)
        
        images = [query_resized]
        for result in results:
            if result.image is not None:
                resized = cv2.resize(result.image, target_size)
                images.append(resized)
        
        # Create grid
        n_cols = min(3, len(images))
        n_rows = (len(images) + n_cols - 1) // n_cols
        
        grid = np.zeros((n_rows * target_size[0], n_cols * target_size[1], 3), 
                       dtype=np.uint8)
        
        for idx, img in enumerate(images):
            row = idx // n_cols
            col = idx % n_cols
            y_start = row * target_size[0]
            x_start = col * target_size[1]
            grid[y_start:y_start+target_size[0], 
                 x_start:x_start+target_size[1]] = img
        
        cv2.imwrite(output_path, grid)
        print(f"Results saved to {output_path}")


# Convenience functions for backward compatibility
def read_images(folder_path: str) -> List[np.ndarray]:
    """Load images from folder (legacy function)."""
    system = CBIRSystem()
    return system.load_images(folder_path)


def show_top_matches(query_img: np.ndarray, 
                    distances: np.ndarray,
                    database_imgs: List[np.ndarray],
                    top_n: int = 5) -> None:
    """Display top matches (legacy function)."""
    top_indices = np.argsort(distances)[:top_n]
    results = []
    for idx in top_indices:
        results.append(SearchResult(
            index=int(idx),
            distance=float(distances[idx]),
            image=database_imgs[idx] if idx < len(database_imgs) else None
        ))
    CBIRVisualizer.display_results(query_img, results)


""" if __name__ == "__main__":
    # Example usage
    print("CBIR System Example")
    print("=" * 50)
    
    # Initialize system
    cbir = CBIRSystem(descriptor_type=DescriptorType.FULL)
    
    # Index database (uncomment and adjust path)
    # cbir.index_database("CBIR_DataSet/obj_decoys")
    
    # Load query image (uncomment and adjust path)
    # query_img = cv2.imread("CBIR_DataSet/img_requetes/query.jpg")
    
    # Search
    # results = cbir.search(query_img, top_n=5)
    
    # Visualize
    # CBIRVisualizer.display_results(query_img, results)
    
    print("Module loaded successfully!") """
