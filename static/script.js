const fileInput = document.getElementById("file");
const fileName = document.getElementById("file-name");

if (fileInput && fileName) {
    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            fileName.textContent = fileInput.files[0].name;
        } else {
            fileName.textContent =
                "PDF, PNG, JPG, JPEG, WEBP or BMP";
        }
    });
}


function copyCaption() {
    const caption = document.querySelector(".improved-caption");

    if (!caption) {
        return;
    }

    navigator.clipboard.writeText(
        caption.innerText
    ).then(() => {
        const button = document.querySelector(".copy-button");

        if (!button) {
            return;
        }

        const originalText = button.textContent;

        button.textContent = "Copied!";

        setTimeout(() => {
            button.textContent = originalText;
        }, 1500);
    });
}