
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from pathlib import Path

from cbir_system import (
    CBIRSystem,
    DescriptorType,
    CBIRVisualizer,
    ImageTransformer,
    SearchResult,
)


# Page configuration
st.set_page_config(
    page_title="Content-Based Image Retrieval",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main-header { 
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 1rem;
    }
    .result-card {
        border: 2px solid #E0E0E0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-size: 1rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Session state initialization
if "cbir_system" not in st.session_state:
    st.session_state.cbir_system = None
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "query_image" not in st.session_state:
    st.session_state.query_image = None


def convert_cv2_to_pil(cv2_img):
    """Convert OpenCV image (BGR) to PIL Image (RGB)."""
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))


def convert_pil_to_cv2(pil_img):
    """Convert PIL Image (RGB) to OpenCV image (BGR)."""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def main():
    # Header
    st.markdown(
        '<h1 class="main-header">Content-Based Image Retrieval</h1>', unsafe_allow_html=True
    )
    st.markdown(
        "**Content-Based Image Retrieval** - Find similar images using "
        "color, histogram, and texture features"
    )

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Dataset path
        dataset_path = st.text_input(
            "Dataset Path",
            value="CBIR_DataSet/obj_decoys",
            help="Path to the folder containing database images",
        )

        # Descriptor type
        descriptor_option = st.selectbox(
            "Descriptor Type",
            options=[
                "Color Moments Only (6D)",
                "Color Histogram (30D)",
                "Texture (34D)",
                "Full: Color + Histogram + Texture (34D)"
            ],
            index=2,
            help="Choose which features to extract from images",
        )

        # Map selection to DescriptorType
        descriptor_map = {
            "Color Moments Only (6D)": DescriptorType.COLOR_ONLY,
            "Color Histogram (30D)": DescriptorType.COLOR_HISTOGRAM,
            "Full: Color + Histogram + Texture (34D)": DescriptorType.FULL,
        }
        descriptor_type = descriptor_map[descriptor_option]

        # Histogram bins
        bins = st.slider(
            "Histogram Bins",
            min_value=4,
            max_value=16,
            value=8,
            step=2,
            help="Number of bins for histogram quantization",
        )

        # Number of results
        top_n = st.slider(
            "Number of Results",
            min_value=1,
            max_value=20,
            value=5,
            help="How many similar images to retrieve",
        )

        st.markdown("---")

        # Index database button
        if st.button("📚 Index Database", key="index_btn"):
            if not os.path.exists(dataset_path):
                st.error(f"❌ Path not found: {dataset_path}")
            else:
                with st.spinner("Indexing database... This may take a moment."):
                    try:
                        st.session_state.cbir_system = CBIRSystem(
                            descriptor_type=descriptor_type, bins=bins
                        )
                        st.session_state.cbir_system.index_database(
                            dataset_path, verbose=False
                        )
                        st.session_state.indexed = True
                        st.success(
                            f"✅ Indexed {len(st.session_state.cbir_system.images)} images!"
                        )
                    except Exception as e:
                        st.error(f"Error indexing database: {str(e)}")

        # Display indexing status
        if st.session_state.indexed:
            st.info(
                f"✓ Database indexed: {len(st.session_state.cbir_system.images)} images"
            )
        else:
            st.warning("⚠️ Database not indexed yet")

        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            """
        This CBIR system uses:
        - **Color Moments**: Mean & std of RGB channels
        - **HSV Histogram**: Color distribution
        - **GLCM Texture**: Contrast, correlation, energy, homogeneity
        """
        )

    # Main content area
    tab1, tab2, tab3 = st.tabs(["🔎 Search", "🔄 Transformations", "📊 Statistics"])

    with tab1:
        search_tab(top_n, dataset_path)

    with tab2:
        transformations_tab(top_n)

    with tab3:
        statistics_tab()


def search_tab(top_n, dataset_path):
    """Main search functionality tab."""
    st.markdown(
        '<h2 class="sub-header">Upload Query Image</h2>', unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Upload a query image to find similar images",
        )

        if uploaded_file is not None:
            # Convert uploaded file to OpenCV format
            pil_image = Image.open(uploaded_file)
            query_img = convert_pil_to_cv2(pil_image)
            st.session_state.query_image = query_img

            # Display query image
            st.image(pil_image, caption="Query Image", use_container_width=True)

            # Image info
            st.caption(f"Size: {query_img.shape[1]}x{query_img.shape[0]} px")

            # Search button
            if st.button("🔍 Search Similar Images", key="search_btn"):
                if not st.session_state.indexed:
                    st.error("❌ Please index the database first (see sidebar)")
                else:
                    with st.spinner("Searching for similar images..."):
                        try:
                            results = st.session_state.cbir_system.search(
                                query_img, top_n=top_n
                            )
                            st.session_state.search_results = results
                            st.success(f"✅ Found {len(results)} similar images!")
                        except Exception as e:
                            st.error(f"Search error: {str(e)}")

    with col2:
        if st.session_state.search_results:
            st.markdown(
                '<h2 class="sub-header">Search Results</h2>', unsafe_allow_html=True
            )

            display_results(st.session_state.search_results)


def transformations_tab(top_n):
    """Geometric transformations and robustness testing tab."""
    st.markdown(
        '<h2 class="sub-header">Test Robustness with Transformations</h2>',
        unsafe_allow_html=True,
    )

    if st.session_state.query_image is None:
        st.info("👆 Please upload a query image in the Search tab first")
        return

    if not st.session_state.indexed:
        st.warning("⚠️ Please index the database first (see sidebar)")
        return

    st.markdown("Apply geometric transformations to test CBIR robustness:")

    # Transformation selection
    transform_type = st.selectbox(
        "Select Transformation",
        [
            "Rotation",
            "Scaling",
            "Translation",
            "Flip Horizontal",
            "Flip Vertical",
            "Add Noise",
            "Brightness Adjustment",
        ],
    )

    # Parameters based on transformation
    col1, col2 = st.columns([1, 1])

    with col1:
        if transform_type == "Rotation":
            angle = st.slider("Rotation Angle (degrees)", -180, 180, 45)
            transformed = ImageTransformer.rotate(st.session_state.query_image, angle)

        elif transform_type == "Scaling":
            scale = st.slider("Scale Factor", 0.1, 3.0, 0.5, 0.1)
            transformed = ImageTransformer.scale(st.session_state.query_image, scale)

        elif transform_type == "Translation":
            tx = st.slider("Horizontal Translation (px)", -100, 100, 20)
            ty = st.slider("Vertical Translation (px)", -100, 100, 20)
            transformed = ImageTransformer.translate(
                st.session_state.query_image, tx, ty
            )

        elif transform_type == "Flip Horizontal":
            transformed = ImageTransformer.flip(st.session_state.query_image, 1)

        elif transform_type == "Flip Vertical":
            transformed = ImageTransformer.flip(st.session_state.query_image, 0)

        elif transform_type == "Add Noise":
            noise_level = st.slider("Noise Level", 0.0, 1.0, 0.1, 0.05)
            transformed = ImageTransformer.add_noise(
                st.session_state.query_image, var=noise_level
            )

        elif transform_type == "Brightness Adjustment":
            brightness = st.slider("Brightness Factor", 0.1, 3.0, 1.2, 0.1)
            transformed = ImageTransformer.adjust_brightness(
                st.session_state.query_image, brightness
            )

        # Display transformed image
        st.image(
            convert_cv2_to_pil(transformed),
            caption=f"Transformed Image ({transform_type})",
            use_container_width=True,
        )

        # Search with transformed image
        if st.button("🔍 Search with Transformed Image", key="transform_search"):
            with st.spinner("Searching..."):
                try:
                    results = st.session_state.cbir_system.search(
                        transformed, top_n=top_n
                    )
                    st.session_state.search_results = results
                    st.success("✅ Search completed!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    with col2:
        if st.session_state.search_results:
            st.markdown("**Results:**")
            display_results(st.session_state.search_results, compact=True)


def statistics_tab():
    """Display system statistics and information."""
    st.markdown('<h2 class="sub-header">System Statistics</h2>', unsafe_allow_html=True)

    if not st.session_state.indexed:
        st.info("Index a database to see statistics")
        return

    cbir = st.session_state.cbir_system

    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Images", len(cbir.images))

    with col2:
        st.metric("Feature Dimensions", cbir.descriptor_type.value)

    with col3:
        if cbir.images:
            avg_size = np.mean([img.shape[0] * img.shape[1] for img in cbir.images])
            st.metric("Avg Image Size", f"{int(avg_size):,} px")

    with col4:
        st.metric("Index Size", f"{cbir.index_matrix.nbytes / 1024:.1f} KB")

    st.markdown("---")

    # Feature distribution
    if cbir.index_matrix is not None:
        st.markdown("### Feature Statistics")

        feature_stats = {
            "Mean": np.mean(cbir.index_matrix, axis=0),
            "Std Dev": np.std(cbir.index_matrix, axis=0),
            "Min": np.min(cbir.index_matrix, axis=0),
            "Max": np.max(cbir.index_matrix, axis=0),
        }

        import pandas as pd

        df = pd.DataFrame(feature_stats)
        df.index = [f"Feature {i+1}" for i in range(len(df))]

        st.dataframe(df, use_container_width=True)

        # Feature visualization
        st.markdown("### Feature Distribution")
        st.line_chart(cbir.index_matrix.T)

    # Dataset info
    st.markdown("---")
    st.markdown("### Dataset Information")

    if cbir.image_paths:
        st.write(f"**Database Path:** {os.path.dirname(cbir.image_paths[0])}")

        # Sample images
        st.markdown("**Sample Images from Database:**")
        cols = st.columns(5)
        for i, col in enumerate(cols):
            if i < len(cbir.images):
                with col:
                    st.image(
                        convert_cv2_to_pil(cbir.images[i]),
                        use_container_width=True,
                        caption=f"Image {i+1}",
                    )


def display_results(results, compact=False):
    """Display search results in a grid."""
    if not results:
        st.info("No results to display")
        return

    # Grid layout
    n_cols = 3 if not compact else 2

    for i in range(0, len(results), n_cols):
        cols = st.columns(n_cols)

        for j, col in enumerate(cols):
            if i + j < len(results):
                result = results[i + j]

                with col:
                    if result.image is not None:
                        st.image(
                            convert_cv2_to_pil(result.image),
                            use_container_width=True,
                            caption=f"Rank {i+j+1}",
                        )
                        st.caption(f"Distance: {result.distance:.4f}")
                        st.caption(f"Index: {result.index}")
                    else:
                        st.warning(f"Image {result.index} not available")


if __name__ == "__main__":
    main()
