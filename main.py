import cv2
import os
import json
import argparse

COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (255, 255, 0),
    (255, 0, 255),
    (147, 20, 255),
    (42, 42, 165),
    (128, 128, 128),
    (128, 0, 96),
]


class TuSimpleLabeler:
    def __init__(self, args):
        self.image_path = args.image_path
        self.label_path = os.path.join(self.image_path, "train_gt.json")
        self.image_files = self.get_image_files()
        self.n_images = len(self.image_files)
        self.labels = self.get_labels()
        if not os.path.isfile(os.path.join(self.image_path, '.cache')):
            self.current_image_index = -1
        else:
            with open(os.path.join(self.image_path, '.cache'), 'r') as f:
                self.current_image_index = int(f.read()) - 1

        self.x = 0
        self.y = 0

        self.start_y = 240
        self.end_y = 720
        self.step = 10
        self.h_samples = list(range(self.start_y, self.end_y, self.step))

        self.width = 1280
        self.height = 720

        self.circle_radius = 7

        self.lane_idx = 0
        self.max_lane_idx = 0
        self.circles = {self.lane_idx: []}
        self.lines = [
            ((0, y), (self.width - 1, y))
            for y in range(self.start_y, self.end_y, self.step)
        ]
        self.window_name = "Circle Drawer"
        self.legend = cv2.imread("legend.png")
        self.legend_height, self.legend_width, _ = self.legend.shape
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def get_image_files(self):
        # 이미지 파일 목록 가져오기
        image_files = []
        for file in os.listdir(self.image_path):
            if file.endswith(".jpg") or file.endswith(".png"):
                image_files.append(file)

        return image_files

    def get_labels(self):
        if not os.path.isfile(self.label_path):
            with open(self.label_path, "w") as _:
                pass
            return {}
        labels = [json.loads(line) for line in open(self.label_path, "r")]
        labels = {line["raw_file"]: line["lanes"] for line in labels}
        return labels

    def show_next_image(self, is_next):
        if is_next:
            self.current_image_index += 1
        else:
            if self.current_image_index == 0:
                return True
            self.current_image_index -= 1

        if self.current_image_index >= len(self.image_files):
            print("All images labeled.")
            return False

        file_name = self.image_files[self.current_image_index]

        self.image = cv2.imread(os.path.join(self.image_path, file_name))
        self.image[
            10 : 10 + self.legend_height :, -10 - self.legend_width : -10
        ] = self.legend
        cv2.putText(self.image, f"{self.current_image_index + 1} / {self.n_images}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imshow(self.window_name, self.image)

        if not file_name in self.labels.keys():
            self.labels[file_name] = []
            self.lane_idx = 0
            self.max_lane_idx = 0
            self.circles = {self.lane_idx: []}
        else:
            self.lane_idx = len(self.labels[file_name])
            self.max_lane_idx = max(self.max_lane_idx, self.lane_idx)
            self.circles = {}
            for i, lane in enumerate(self.labels[file_name]):
                self.circles[i] = [
                    (x, y) for x, y in zip(lane, self.h_samples) if x > 0
                ]
            self.circles[self.lane_idx] = []

        return True

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.circles[self.lane_idx].append((self.x, self.y))
            self.draw_lines_circles()
        elif event == cv2.EVENT_MOUSEMOVE:
            self.x = x
            nearest_y = (
                self.start_y
                if y < self.start_y
                else self.end_y
                if y > self.end_y
                else round(y, -1)
            )
            self.y = nearest_y
            self.draw_lines_circles()

    def draw_lines_circles(self):
        image_copy = self.image.copy()
        for lane_idx in range(0, self.max_lane_idx + 1):
            for circle in self.circles.get(lane_idx, []):
                cv2.circle(
                    image_copy,
                    circle,
                    self.circle_radius,
                    COLORS[lane_idx % len(COLORS)],
                    thickness=2,
                )
        cv2.circle(
            image_copy,
            (self.x, self.y),
            self.circle_radius,
            COLORS[self.lane_idx % len(COLORS)],
            thickness=2,
        )

        for start, end in self.lines:
            cv2.line(image_copy, start, end, (0, 0, 255), 1)
        cv2.imshow(self.window_name, image_copy)

    def cancel_last_circle(self):
        if self.circles[self.lane_idx]:
            self.circles[self.lane_idx].pop()

    def save_circles(self):
        with open(self.label_path, "r") as file:
            lines = file.readlines()

        labels = []
        for i, circles in self.circles.items():
            if not circles:
                continue
            tmp_labels = [-2] * ((self.end_y - self.start_y) // self.step)
            for x, y in circles:
                tmp_labels[(y - self.start_y) // self.step] = x
            labels.append(tmp_labels)

        file_name = self.image_files[self.current_image_index]
        for i, line in enumerate(lines):
            line = json.loads(line)
            if line["raw_file"] == file_name:
                line["lanes"] = labels
                lines[i] = json.dumps(line) + "\n"
                break
        else:
            file_string = {
                "lanes": labels,
                "h_samples": self.h_samples,
                "raw_file": file_name,
            }
            lines += json.dumps(file_string) + "\n"

        # 수정된 내용으로 파일 다시 쓰기
        with open(self.label_path, "w") as file:
            file.writelines(lines)

        self.labels[file_name] = labels

        self.lane_idx += 1
        self.max_lane_idx = max(self.max_lane_idx, self.lane_idx)
        self.circles[self.lane_idx] = self.circles.get(self.lane_idx, [])

    def change_lane(self, is_next):
        if is_next:
            self.lane_idx = min(self.lane_idx + 1, self.max_lane_idx)
        else:
            self.lane_idx = max(self.lane_idx - 1, 0)

    def run(self):
        self.show_next_image(is_next=True)
        self.draw_lines_circles()

        while True:
            key = cv2.waitKeyEx()
            if key == 127:  # delete key
                self.cancel_last_circle()
            elif key == 13:  # Enter key
                self.save_circles()
            elif key == 27:  # ESC key
                self.save_circles()
                with open(os.path.join(self.image_path, '.cache'), 'w') as f:
                    f.write(str(self.current_image_index))
                break
            elif key == 63235:  # right button
                self.save_circles()
                if not self.show_next_image(is_next=True):
                    break
            elif key == 63234:  # left button
                self.save_circles()
                self.show_next_image(is_next=False)
            elif key in [122, 90, 12619]:  # z
                self.change_lane(is_next=False)
            elif key in [120, 88, 12620]:  # x
                self.change_lane(is_next=True)
            self.draw_lines_circles()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--image_path", type=str, default="dataset", help="path of image dataset"
    )
    args = parser.parse_args()

    labeler = TuSimpleLabeler(args)
    labeler.run()
