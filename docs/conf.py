# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = "HEonGPU"
copyright = "2025, Ali Şah Özcan"
author = "Ali Şah Özcan"
release = "2.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
    "breathe",
    "exhale",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_extra_path = ["_extra"]

# -- Breathe Configuration -------------------------------------------------
breathe_projects = {"HEonGPU": "doxygen/xml/"}
breathe_default_project = "HEonGPU"

# 3. Exhale Configuration
# This is where the automation is defined.
exhale_args = {
    # --- Core Exhale Settings ---
    "containmentFolder":      "./full_api",
    "rootFileName":           "library_root.rst",
    "rootFileTitle":          "Full API Listing",
    "createTreeView":         True,

    # --- Doxygen Execution Control ---
    "exhaleExecutesDoxygen": True,
    "doxygenStripFromPath":   "..",
    "exhaleDoxygenStdin":  """
        # Doxyfile settings extracted for Exhale
        PROJECT_NAME           = "HEonGPU"
        PROJECT_BRIEF          = "HEonGPU is a high-performance library that optimizes Fully Homomorphic Encryption (FHE) on GPUs."
        
        # Input & Pathing
        # IMPORTANT: Verify 'INPUT' paths are correct relative to the 'docs' directory.
        INPUT                  = ../src
        RECURSIVE              = YES
        FULL_PATH_NAMES        = YES

        # File Patterns & Mappings for C++/CUDA
        FILE_PATTERNS          = *.cu *.cuh *.h *.hpp *.c *.cpp
        EXTENSION_MAPPING      = cu=C++ cuh=C++

        # Preprocessor settings for CUDA keywords
        ENABLE_PREPROCESSING   = YES
        PREDEFINED             = __global__= \\
                                 __device__= \\
                                 __host__= \\
                                 __shared__= \\
                                 __constant__=

        # Extraction Rules
        EXTRACT_ALL            = NO
        EXTRACT_PRIVATE        = NO
        EXTRACT_STATIC         = NO
        BRIEF_MEMBER_DESC      = YES
        REPEAT_BRIEF           = YES

        # XML Generation for Breathe/Exhale
        GENERATE_XML           = YES
        XML_PROGRAMLISTING     = YES
    """
}

# Tell sphinx what the primary language being documented is.
primary_domain = "cpp"

# Tell sphinx what the C++ syntax highlighting scheme to use is.
highlight_language = "cu"


