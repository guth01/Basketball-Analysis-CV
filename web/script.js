const form = document.querySelector("#uploadForm");
const input = document.querySelector("#videoInput");
const dropZone = document.querySelector("#dropZone");
const fileName = document.querySelector("#fileName");
const processButton = document.querySelector("#processButton");
const statusCard = document.querySelector("#statusCard");
const statusTitle = document.querySelector("#statusTitle");
const statusText = document.querySelector("#statusText");
const resultCard = document.querySelector("#resultCard");
const resultName = document.querySelector("#resultName");
const downloadLink = document.querySelector("#downloadLink");

function setSelectedFile(file) {
  if (!file) {
    fileName.textContent = "MP4, AVI, MOV, MKV, or WEBM";
    processButton.disabled = true;
    return;
  }

  fileName.textContent = file.name;
  processButton.disabled = false;
  resultCard.classList.add("hidden");
}

function showStatus(title, text, isError = false) {
  statusCard.classList.remove("hidden");
  statusTitle.textContent = title;
  statusText.textContent = text;
  statusText.classList.toggle("error", isError);
}

input.addEventListener("change", () => {
  setSelectedFile(input.files[0]);
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("drag-over");

  const [file] = event.dataTransfer.files;
  if (!file) {
    return;
  }

  const transfer = new DataTransfer();
  transfer.items.add(file);
  input.files = transfer.files;
  setSelectedFile(file);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = input.files[0];
  if (!file) {
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  processButton.disabled = true;
  resultCard.classList.add("hidden");
  showStatus("Processing video", "The pipeline is running. This can take a while for longer clips.");

  try {
    const response = await fetch("/api/process", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      const message = typeof payload.detail === "string"
        ? payload.detail
        : payload.detail?.message || "The pipeline failed while processing this video.";
      throw new Error(message);
    }

    downloadLink.href = payload.download_url;
    resultName.textContent = payload.filename;
    resultCard.classList.remove("hidden");
    statusCard.classList.add("hidden");
  } catch (error) {
    showStatus("Processing failed", error.message, true);
  } finally {
    processButton.disabled = false;
  }
});
