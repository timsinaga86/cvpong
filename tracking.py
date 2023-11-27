import os
import cv2
import numpy as np

def image_matching(template, search_region, method):
    template_resized = cv2.resize(template, (search_region.shape[1], search_region.shape[0]))

    if method == 'sum_squared_difference':
        result = np.sum(np.square(template_resized - search_region))
    elif method == 'cross_correlation':
        result = np.sum(template_resized * search_region)
    elif method == 'normalized_cross_correlation':
        result = np.sum(template_resized * search_region) / (np.sqrt(np.sum(template_resized**2)) * np.sqrt(np.sum(search_region**2)))
    else:
        raise ValueError("Invalid matching method.")

    return result

def local_exhaustive_search(frame, template, bbox, method):
    if len(bbox) == 2:
        x, y = bbox
        w, h = 40, 40
    elif len(bbox) == 4:
        x, y, w, h = bbox
    else:
        raise ValueError("Invalid bounding box format.")
    search_region = frame[y-1:y+h+1, x-1:x+w+1]

    min_score = np.inf
    best_match = (x, y)

    for i in range(-10, 11):
        for j in range(-10, 11):
            shifted_region = frame[y-1+j:y+h+1+j, x-1+i:x+w+1+i]

            if shifted_region.shape == search_region.shape:
                score = image_matching(template, shifted_region, method)
                if score < min_score:
                    min_score = score
                    best_match = (x+i, y+j)

    return best_match

def draw_bounding_box(frame, bbox, save_path, frame_num):
    if len(bbox) == 2:
        x, y = bbox
        w, h = 40, 40
    elif len(bbox) == 4:
        x, y, w, h = bbox
    else:
        raise ValueError("Invalid bounding box format.")
    print(bbox)
    x = max(0, x)
    y = max(0, y)
    w = min(frame.shape[1] - x, w)
    h = min(frame.shape[0] - y, h)

    frame_with_bbox = frame.copy()
    cv2.rectangle(frame_with_bbox, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imwrite(os.path.join(save_path, f'{frame_num:04d}.jpg'), frame_with_bbox)
    cv2.imshow('Frame with Bounding Box', frame_with_bbox)
    cv2.waitKey(1)


def main():
    images_folder = 'image_girl/image_girl'
    result_folder = 'result_folder_ssd'
    output_path = 'output_video_ssd.avi'
    x = 45
    y = 20
    width = 50
    height = 50
    method = 'sum_squared_difference'

    os.makedirs(result_folder, exist_ok=True)

    frame = cv2.imread(os.path.join(images_folder, '0001.jpg'))
    bbox = (x, y, width, height)
    draw_bounding_box(frame, bbox, result_folder, 1)
    frame_width, frame_height = frame.shape[1], frame.shape[0]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), 10, (frame_width, frame_height))

    # Iterate through frames
    for frame_num in range(2, 501):
        frame = cv2.imread(os.path.join(images_folder, f'{frame_num:04d}.jpg'))
        new_bbox = local_exhaustive_search(frame, frame[y:y+height, x:x+width], bbox, method)
        draw_bounding_box(frame, new_bbox, result_folder, frame_num)

        bbox = new_bbox

        out.write(frame)

    out.release()
    cv2.destroyAllWindows() 

    fps = 24
    frame_size = (128, 96)
    images = [img for img in os.listdir(result_folder) if img.endswith(".jpg")]
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size)

    for image in images:
        img_path = os.path.join(result_folder, image)
        frame = cv2.imread(img_path)
        video_writer.write(frame)

    video_writer.release()
    print("Processing complete. Output video saved as:", output_path)

if __name__ == "__main__":
    main()
