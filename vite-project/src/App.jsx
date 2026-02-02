import { useEffect, useRef, useState } from "react";
import "./App.css";
import Webcam from "react-webcam";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";

let interval;
let lastAlertTime = 0; // spam control

const App = () => {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    startPrediction();

    return () => {
      if (interval) clearInterval(interval);
    };
  }, []);

  const startPrediction = async () => {
    await tf.ready();
    await tf.setBackend("webgl"); // GPU

    const model = await cocoSsd.load();
    setLoading(false);

    interval = setInterval(() => {
      detect(model);
    }, 200); // 200ms is stable
  };

  const detect = async (model) => {
    if (
      webcamRef.current &&
      webcamRef.current.video &&
      webcamRef.current.video.readyState === 4
    ) {
      const video = webcamRef.current.video;
      const videoWidth = video.videoWidth;
      const videoHeight = video.videoHeight;

      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;

      const predictions = await model.detect(video);
      const ctx = canvasRef.current.getContext("2d");

      drawMesh(predictions, ctx);
      handleCounts(predictions);
    }
  };

  const drawMesh = (predictions, ctx) => {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    predictions.forEach((prediction) => {
      const [x, y, width, height] = prediction.bbox;
      const text = prediction.class;

      ctx.beginPath();
      ctx.strokeStyle = "green";
      ctx.lineWidth = 2;
      ctx.rect(x, y, width, height);
      ctx.stroke();

      ctx.font = "18px Arial";
      ctx.fillStyle = "green";
      ctx.fillText(text, x, y - 5);
    });
  };

  const handleCounts = (predictions) => {
    let personCount = 0;
    let phoneCount = 0;

    predictions.forEach((p) => {
      if (p.class === "person") personCount++;
      if (p.class === "cell phone") phoneCount++;
    });

    const now = Date.now();

    // alert only once in 3 seconds
    if (now - lastAlertTime > 3000) {
      if (personCount > 1) {
        alert("⚠️ More than one person detected");
        lastAlertTime = now;
      }

      if (phoneCount > 1) {
        alert("📱 More than one phone detected");
        lastAlertTime = now;
      }
    }
  };

  return (
    <div className="parentsContainer">
      <h1 className="appTitle">🎯 Real-time Object Detection</h1>
      {loading && <span>Loading Model.....</span>}
      <div className="videoWrapper">
        <Webcam ref={webcamRef} />
        <canvas ref={canvasRef} />
      </div>
    </div>
  );
};

export default App;