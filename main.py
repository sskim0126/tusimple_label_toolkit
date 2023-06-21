import cv2
import os
import json
import argparse

COLORS = [
    (0, 255, 0),
    (255, 0, 0)
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
        self.max_lane_idx = len(COLORS) - 1
        self.circles = {key: [] for key in range(self.max_lane_idx + 1)}
        self.lines = [
            ((0, y), (self.width - 1, y))
            for y in range(self.start_y, self.end_y, self.step)
        ]
        self.straight_mode = True

        self.window_name = "Circle Drawer"
        self.legend = cv2.imread("legend.png")
        self.legend_height, self.legend_width, _ = self.legend.shape
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def get_image_files(self):
        # 이미지 파일 목록 가져오기
        image_files = []
        for file in sorted(os.listdir(self.image_path)):
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
        cv2.putText(self.image, f"{self.current_image_index + 1} / {self.n_images}, straight mode: {self.straight_mode}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imshow(self.window_name, self.image)

        if not file_name in self.labels.keys():
            self.labels[file_name] = []
            self.lane_idx = 0
            self.circles = {key: [] for key in range(self.max_lane_idx + 1)}
        else:
            self.lane_idx = len(self.labels[file_name]) - 1
            self.circles = {key: [] for key in range(self.max_lane_idx + 1)}
            for i, lane in enumerate(self.labels[file_name]):
                self.circles[i] = [
                    (x, y) for x, y in zip(reversed(lane), reversed(self.h_samples)) if x > 0
                ]

        return True

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.circles[self.lane_idx].append((self.x, self.y))
            if self.straight_mode and len(self.circles[self.lane_idx]) >= 2:
                last = self.circles[self.lane_idx][-2]
                gap = abs(last[1] - self.y)
                if gap > self.step:
                    slope = (last[0] - self.x) / (last[1] - self.y)
                    if last[1] > self.y:
                        for i, middel_y in enumerate(range(self.y + self.step, last[1], self.step), 1):
                            self.circles[self.lane_idx].append((round(self.x + slope * i * self.step), middel_y))
                    else:
                        for i, middel_y in enumerate(range(last[1] + self.step, self.y, self.step), 1):
                            self.circles[self.lane_idx].append((round(last[0] + slope * i * self.step), middel_y))
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

        tmp_circles = self.circles.get(self.lane_idx, [])
        if len(tmp_circles) >= 2:
            # 기울기와 y 절편 계산
            start_point = tmp_circles[-2]
            end_point = tmp_circles[-1]

            if end_point[0] != start_point[0]:
                slope = (end_point[1] - start_point[1]) / (end_point[0] - start_point[0])
                intercept = start_point[1] - slope * start_point[0]

                # 직선 그리기
                x1 = 0
                y1 = int(slope * x1 + intercept)

                x2 = self.width - 1
                y2 = int(slope * x2 + intercept)

                cv2.line(image_copy, (x1, y1), (x2, y2), (255, 255, 255), 1)
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

        self.change_lane(is_next=True)
        self.circles[self.lane_idx] = self.circles.get(self.lane_idx, [])

    def change_lane(self, is_next):
        if is_next:
            self.lane_idx = min(self.lane_idx + 1, self.max_lane_idx)
        else:
            self.lane_idx = max(self.lane_idx - 1, 0)
    
    # lane이 2개일때만 작동
    # 수정해야함
    def swap_lane(self):
        self.circles[0], self.circles[1] = self.circles[1], self.circles[0]
        self.lane_idx = 1 - self.lane_idx

    def run(self):
        self.show_next_image(is_next=True)
        self.draw_lines_circles()

        while True:
            key = cv2.waitKeyEx()
            if key == 127 or key == 8:  # delete key
                self.cancel_last_circle()
            elif key == 13:  # Enter key
                self.save_circles()
            elif key == 27:  # ESC key
                self.save_circles()
                with open(os.path.join(self.image_path, '.cache'), 'w') as f:
                    f.write(str(self.current_image_index))
                break
            elif key == 63235 or key == 2555904:  # right button
                self.save_circles()
                if not self.show_next_image(is_next=True):
                    break
            elif key == 63234 or key == 2424832:  # left button
                self.save_circles()
                self.show_next_image(is_next=False)
            elif key in [122, 90, 12619]:  # z
                self.change_lane(is_next=False)
            elif key in [120, 88, 12620]:  # x
                self.change_lane(is_next=True)
            elif key in [115, 83, 12596]:  # s
                self.straight_mode = not self.straight_mode
                file_name = self.image_files[self.current_image_index]

                self.image = cv2.imread(os.path.join(self.image_path, file_name))
                self.image[
                    10 : 10 + self.legend_height :, -10 - self.legend_width : -10
                ] = self.legend
                cv2.putText(self.image, f"{self.current_image_index + 1} / {self.n_images}, straight mode: {self.straight_mode}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.imshow(self.window_name, self.image)
            elif key in [113, 81, 12610]: # q
                self.circles[self.lane_idx] = []
            elif key in [114, 82, 12593]: # r
                self.swap_lane()

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