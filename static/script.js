document.getElementById("uploadForm").onsubmit = function() {
    let fileInput = document.getElementById("fileInput");
    if (!fileInput.value) {
        alert("Please select an image to upload!");
        return false;
    }
};
