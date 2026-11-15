import cv2
import torch

from inference import infer_image, load_models


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    deepc, refinenet = load_models(
        "./reference/longrun-epoch=99-step=369700.ckpt",
        "./reference/second-refinenet-epoch-100-step=373k.ckpt",
        n_ids=16,
        device=device,
    )

    # Camera 0 is usually the default webcam. Try 1 if you have multiple cameras.
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError(
            "Could not open the webcam. Check camera permissions, "
            "or try changing VideoCapture(0) to VideoCapture(1)."
        )

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a frame from the webcam.")
                break

            # Use the model's training input size (width=320, height=240).
            frame = cv2.resize(frame, (320, 240))

            keypoints, output = infer_image(
                frame,
                16,
                deepc,
                refinenet,
                draw_pred=True,
                device=device,
            )

            cv2.putText(
                output,
                f"Corners: {len(keypoints)}  Device: {device}  (q to quit)",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("DeepCharuco webcam", output)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()