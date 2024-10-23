import cv2
import os
import numpy as np
from pupil_apriltags import Detector


def detect_apriltags(image_path):
    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    results = detector.detect(gray)

    for r in results:

        tag_id = r.tag_id
        center = tuple(int(c) for c in r.center)
        corners = np.array(r.corners, dtype=np.int32)

        cv2.polylines(image, [corners], True, (0, 255, 0), 2)

        cv2.circle(image, center, 5, (0, 0, 255), -1)

        cv2.putText(
            image,
            f"ID: {tag_id}",
            (center[0] - 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return image, len(results)


def main():

    image_folder = "images"

    valid_extensions = (".jpg", ".jpeg", ".png")

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(valid_extensions):
            image_path = os.path.join(image_folder, filename)

            processed_image, num_tags = detect_apriltags(image_path)

            print(f"Found {num_tags} AprilTags in {filename}")

            cv2.imshow(f"AprilTags - {filename}", processed_image)
            cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
