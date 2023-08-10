import cv2

def count_frames_with_green_in_custom_region(video_path, area_threshold):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    green_lower = (154, 65, 56)  # 緑色の下限 (H, S, V)
    green_upper = (140, 32, 30)  # 緑色の上限 (H, S, V)

    # green_upper = (59, 175, 117)
    # green_lower = (11, 33, 22)
    frames_with_green = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        pixel_per_cm_width = width / 21.27  # 画面の幅を基準にして5cmに相当するピクセル数を計算（21.27は5cmに相当するピクセル数）
        pixel_per_cm_height = height / 29.7  # 画面の高さを基準にして15cmに相当するピクセル数を計算（29.7は15cmに相当するピクセル数）

        region_width_cm = 5  # 5cmの領域を選択
        region_width_pixels = int(region_width_cm * pixel_per_cm_width)

        region_height_cm = 15  # 15cmの領域を選択
        region_height_pixels = int(region_height_cm * pixel_per_cm_height)

        region_of_interest = frame[:region_height_pixels, :region_width_pixels]  # 画面の左上から5cm幅・15cm高さの領域を選択

        # hsv_roi = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2HSV)
        hsv_roi = region_of_interest
        green_mask = cv2.inRange(hsv_roi, green_lower, green_upper)

        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= area_threshold:
                frames_with_green += 1
                break  # マスクされた領域が見つかったら次のフレームへ

        cv2.imshow('Frame', frame)
        cv2.imshow('Region of Interest', region_of_interest)
        # cv2.imshow('Region of Interest', hsv_roi)
        cv2.imshow('Green Mask', green_mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q'キーを押すと終了
            break

    cap.release()
    cv2.destroyAllWindows()

    return frames_with_green

if __name__ == "__main__":
    video_path = '5.mp4'
    area_threshold = 1000  # マスクされた領域の最小面積（適切な値を設定してください）

    frames_with_green = count_frames_with_green_in_custom_region(video_path, area_threshold)
    print("Frames with green in custom region:", frames_with_green)
