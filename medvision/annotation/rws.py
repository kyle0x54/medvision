import json
import medvision as mv


def get_rws_datainfo_path(dcm_path):
    return mv.splitext(dcm_path)[0] + ".datainfo"


def get_rws_annot_path(dcm_path):
    return mv.splitext(dcm_path)[0] + ".json"


def get_rws_flag_path(dcm_path):
    return mv.splitext(dcm_path)[0] + ".flag"


def get_rws_text_path(dcm_path):
    return mv.splitext(dcm_path)[0] + ".text"


def load_rws_contour(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # relative path from label file to relative path from cwd
    image_path = mv.joinpath(mv.parentdir(filepath), data['imagePath'])
    height = data.get('imageHeight')
    width = data.get('imageWidth')
    shapes = []
    for s in data['shapes']:
        shape = (
            s['label'],
            s['points'],
        )
        shapes.append(shape)

    return {
        'height': height,
        'width': width,
        'image_path': image_path,
        'shapes': shapes
    }


def load_rws_bbox(filepath):
    rws_bboxes = load_rws_contour(filepath)

    shapes = []
    for label, points in rws_bboxes['shapes']:
        assert len(points) == 2, 'only support rectangle annotation'
        top_left = points[0]
        bottom_right = points[1]
        shape = (
            label,
            [
                top_left[0],
                top_left[1],
                bottom_right[0],
                bottom_right[1]
            ]
        )
        shapes.append(shape)

    rws_bboxes['shapes'] = shapes

    return rws_bboxes


if __name__ == '__main__':
    bboxes = load_rws_bbox('/home/huiying/Test/test.json')
    print(bboxes)
