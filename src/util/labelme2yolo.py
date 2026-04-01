import json
import os
import argparse


def convert_labelme_to_yolo(labelme_json_path, output_txt_path, class_id=0, double_points=False):
    """
    it maybe a walkaroud to have a copy of keypoints concat behand the orginal one, when try to use yolo flip augmentation while points is not symmetrial. When the  'double_points' is True, points [a,b,c] is double as [a,a,b,b,c,c]
    """
    with open(labelme_json_path, 'r', encoding='utf-8') as f:
        labelme_data = json.load(f)

    # image_filename = labelme_data['imagePath']
    image_height, image_width = labelme_data['imageHeight'], labelme_data['imageWidth']

    anno = labelme_data['shapes']
    instenses = {}
    for shape in anno:
        if shape['shape_type'] != 'point':
            if shape['shape_type'] == "rectangle":
                points = shape["points"] #[[up-left-x,up-left-y],[low-right-x, low-right-y]]
                center_x = (points[0][0]+points[1][0])/2
                center_y = (points[0][1]+points[1][1])/2
                w = abs(points[0][0]-points[1][0])
                h = abs(points[0][1]-points[1][1])
                if shape['group_id']in instenses:
                    instenses[shape['group_id']]['bbox'] = f"{center_x/image_width} {center_y/image_height} {w/image_width} {h/image_height}"
                else:
                    instenses[shape['group_id']] ={
                        'bbox':f"{center_x/image_width} {center_y/image_height} {w/image_width} {h/image_height}"
                    }
            else:
                raise ValueError(f"encountered unsupported shape type {shape['shape_type']} with label {shape['label']}, ar {labelme_json_path}")

        # label = shape['label']
        else:
            x, y = shape['points'][0]
            x_normalized = x / image_width
            y_normalized = y / image_height
            point_str =f"{x_normalized} {y_normalized} {x_normalized} {y_normalized}" if double_points else f"{x_normalized} {y_normalized}"
            if shape['group_id']in instenses:
                if 'points' in instenses[shape['group_id']]:
                    instenses[shape['group_id']]['points'].append(point_str)
                else:
                    instenses[shape['group_id']]['points'] = [point_str]
            else:
                instenses[shape['group_id']] ={
                    'points':[point_str]
                }
    # lines = '\n'.join([' '.join([str(class_id), instence['bbox'],*instence['points']]) for groupid, instence in sorted(instenses.items())])
    lines = '\n'.join([' '.join([str(groupid), instence['bbox'],*instence['points']]) for groupid, instence in sorted(instenses.items())])

    
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(lines)

    print(f"Converted {labelme_json_path} to {output_txt_path}")

def batch_convert_labelme_to_yolo(input_dir, output_dir,double_point):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            labelme_json_path = os.path.join(input_dir, filename)
            output_txt_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.txt')
            convert_labelme_to_yolo(labelme_json_path, output_txt_path,double_points=double_point)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LabelMe JSON files to YOLO format.")
    parser.add_argument("input_dir", help="Directory containing LabelMe JSON files.")
    parser.add_argument("--output_dir", default=None,help="Directory to save YOLO format files. Dafault as input_dir")
    parser.add_argument('-d',"--double_point",action='store_true', help="it maybe a walkaroud to have a copy of keypoints concat behand the orginal one, when try to use yolo flip augmentation while points is not symmetrial. When the  'double_points' is True, points [a,b,c] is double as [a,a,b,b,c,c]")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.input_dir
    batch_convert_labelme_to_yolo(args.input_dir, args.output_dir,args.double_point)