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
        xmin = min(points[0][0], points[1][0])
        ymin = min(points[0][1], points[1][1])
        xmax = max(points[0][0], points[1][0])
        ymax = max(points[0][1], points[1][1])
        shape = (
            label,
            [xmin, ymin, xmax, ymax]
        )
        shapes.append(shape)

    rws_bboxes['shapes'] = shapes

    return rws_bboxes


if __name__ == '__main__':
    bboxes = load_rws_bbox('/home/huiying/Test/test.json')
    print(bboxes)
