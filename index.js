// Paths to the model and metadata
const modelURL = 'models/model.json';
const metadataURL = 'models/metadata.json';

let model;
let labels = [];
const minConfidence = 0.5; // Adjusted confidence threshold

// Load the model and metadata
async function loadModel() {
  console.log("Loading model...");
  model = await tf.loadLayersModel(modelURL);
  console.log("Model loaded successfully.");
}

async function loadMetadata() {
  console.log("Loading metadata...");
  const response = await fetch(metadataURL);
  const metadata = await response.json();
  labels = metadata.labels || [];
  console.log("Metadata loaded:", labels);
}

// Check frame brightness
function isFrameBright(videoElement) {
  const frame = tf.browser.fromPixels(videoElement);
  const grayFrame = frame.mean(2); // Convert to grayscale
  const avgBrightness = grayFrame.mean().dataSync()[0];
  frame.dispose();
  grayFrame.dispose();
  return avgBrightness > 0.1; // Adjust brightness threshold as needed
}

// Detect objects in the current frame
async function detectObjects(videoElement, canvasElement) {
  if (!model) {
    console.error("Model not loaded!");
    return;
  }

  if (!isFrameBright(videoElement)) {
    console.log("Frame is too dark, skipping detection.");
    return;
  }

  const ctx = canvasElement.getContext("2d");
  canvasElement.width = videoElement.videoWidth;
  canvasElement.height = videoElement.videoHeight;

  // Capture video frame and preprocess
  const inputTensor = tf.browser.fromPixels(videoElement)
    .resizeNearestNeighbor([224, 224]) // Resize to model's input size
    .expandDims(0) // Add batch dimension
    .toFloat()
    .div(tf.scalar(255)); // Normalize to [0, 1]

  // Perform prediction
  const predictions = model.predict(inputTensor);
  const predictionArray = await predictions.array();

  // Draw the frame
  ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

  const height = videoElement.videoHeight;
  const width = videoElement.videoWidth;

  const detectedObjects = predictionArray[0]
    .map((confidence, index) => ({ label: labels[index], confidence }))
    .filter(result => result.confidence > minConfidence);

  // Filter detected objects (e.g., person and bottle)
  const detectedClasses = detectedObjects.map(obj => obj.label);
  const isPersonDetected = detectedClasses.includes("person");
  const isBottleDetected = detectedClasses.includes("bottle");

  detectedObjects.forEach((result) => {
    const { label, confidence } = result;

    // NOTE: Replace with actual bounding box coordinates if available
    const upperLeftX = Math.random() * width;
    const upperLeftY = Math.random() * height;
    const lowerRightX = upperLeftX + 100;
    const lowerRightY = upperLeftY + 50;

    // Draw bounding box and label
    ctx.strokeStyle = "red";
    ctx.lineWidth = 2;
    ctx.strokeRect(upperLeftX, upperLeftY, lowerRightX - upperLeftX, lowerRightY - upperLeftY);

    ctx.fillStyle = "red";
    ctx.font = "16px Arial";
    ctx.fillText(
      `${label}: ${(confidence * 100).toFixed(2)}%`,
      upperLeftX,
      upperLeftY > 20 ? upperLeftY - 5 : upperLeftY + 15
    );
  });

  if (isPersonDetected && isBottleDetected) {
    console.log("Person and Bottle detected in the same frame.");
  }

  inputTensor.dispose();
}

// Initialize webcam
async function startWebcam() {
  const video = document.getElementById("webcam");
  const canvas = document.getElementById("outputCanvas");

  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;

  video.addEventListener("loadeddata", () => {
    console.log("Webcam stream started.");
    setInterval(() => detectObjects(video, canvas), 200); // Process every 200ms
  });
}

// Main function
async function main() {
  await loadModel();
  await loadMetadata();
  await startWebcam();
}

// Run the main function
main();
