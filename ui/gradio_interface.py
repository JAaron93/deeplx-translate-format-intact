"""Gradio interface for document translation."""

from pathlib import Path

import gradio as gr

from config.settings import Settings
from core.translation_handler import (
    download_translated_file,
    get_translation_status,
    process_file_upload,
    start_translation,
)


def create_gradio_interface() -> gr.Blocks:
    """Create the advanced Gradio web interface"""
    # Load settings for language configuration
    settings = Settings()

    # Load CSS from external file
    css_path = Path(__file__).parent.parent / "static" / "styles.css"
    try:
        css_content = css_path.read_text()
    except FileNotFoundError:
        # Fallback if CSS file not found
        css_content = ""
        print(f"Warning: CSS file not found at {css_path}")

    with gr.Blocks(
        title="Advanced Document Translator", theme=gr.themes.Soft(), css=css_content
    ) as interface:
        gr.Markdown(
            """
            # 📄 Advanced Document Translator

            Professional document translation with **comprehensive formatting**
            **preservation**.

            🎯 **Features:**
            - Advanced PDF processing with image-text overlay technique
            - Precise text positioning preservation
            - High-resolution rendering (300 DPI)
            - Support for complex layouts and embedded images
            - DOCX and TXT format support
            - Automatic language detection

            📊 **Supported Files:** PDF, DOCX, TXT (up to 10MB)
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # File Upload Section
                gr.Markdown("## 📤 Upload Document")

                file_upload = gr.File(
                    label="Choose File (PDF, DOCX, TXT)",
                    file_types=[".pdf", ".docx", ".txt"],
                    file_count="single",
                    elem_classes=["upload-area"],
                )

                upload_status = gr.Textbox(
                    label="Processing Status", interactive=False, lines=8, max_lines=10
                )

                # Advanced Processing Info
                gr.Markdown("## 🔍 Processing Details")
                processing_info = gr.Textbox(
                    label="Processing Information",
                    interactive=False,
                    lines=8,
                    elem_classes=["info-panel"],
                )

            with gr.Column(scale=2):
                # Preview Section
                gr.Markdown("## 👀 Document Preview")

                document_preview = gr.Textbox(
                    label="Content Preview",
                    lines=12,
                    interactive=False,
                    placeholder=(
                        "Upload a document to see preview with advanced "
                        "processing info..."
                    ),
                )

                # Language and Translation Section
                with gr.Row():
                    with gr.Column():
                        detected_language = gr.Textbox(
                            label="Detected Source Language", interactive=False
                        )

                    with gr.Column():
                        target_language = gr.Dropdown(
                            label="Target Language",
                            choices=settings.SUPPORTED_LANGUAGES,
                            value="English",
                        )

                # Page limit slider (increased from 200 to 2000 pages)
                pages_slider = gr.Slider(
                    minimum=1,
                    maximum=2000,
                    step=1,
                    value=50,
                    label="Pages to translate",
                )

                # Philosophy mode toggle
                philosophy_mode = gr.Checkbox(
                    label="Enable Philosophy Mode (Neologism Detection)", value=False
                )
                # Translation Controls
                translate_btn = gr.Button(
                    "🚀 Start Advanced Translation", variant="primary", size="lg"
                )

                # Progress Section
                gr.Markdown("## 📊 Translation Progress")

                with gr.Row():
                    progress_status = gr.Textbox(
                        label="Status", interactive=False, scale=4
                    )
                    refresh_btn = gr.Button("🔄 Refresh", size="sm", scale=1)

                # Progress bar for visual feedback
                gr.Progress()

                # Export Section
                gr.Markdown("## 💾 Download Translated Document")

                with gr.Row():
                    output_format = gr.Dropdown(
                        label="Output Format",
                        choices=["PDF", "DOCX", "TXT"],
                        value="PDF",
                    )

                    download_btn = gr.Button(
                        "📥 Download", variant="secondary", interactive=False
                    )

                download_file = gr.File(label="Download File", visible=False)

        # Event Handlers
        file_upload.change(
            fn=process_file_upload,
            inputs=[file_upload],
            outputs=[
                document_preview,
                upload_status,
                detected_language,
                processing_info,
            ],
        )

        translate_btn.click(
            fn=start_translation,
            inputs=[target_language, pages_slider, philosophy_mode],
            outputs=[progress_status, upload_status, download_btn],
        )

        # Status update function for manual refresh
        def update_status():
            status, progress, download_ready = get_translation_status()
            return status, gr.update(interactive=download_ready)

        # Connect refresh button to status update
        # Manual refresh
        refresh_btn.click(fn=update_status, outputs=[progress_status, download_btn])

        download_btn.click(
            fn=download_translated_file, inputs=[output_format], outputs=[download_file]
        )

    return interface
