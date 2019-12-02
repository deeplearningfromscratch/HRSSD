import os
import json

label_map = {'bus': 6,
             'person': 15,
             'bike': 2,
             'motor': 14,
             'car': 7,
             'train': 19,
             'rider': 15}

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def dump_json(path, data):
    with open(path, 'w') as f:
        return json.dump(data, f)


def parse_annotation(data):
    boxes = list()
    labels = list()

    for obj in data['labels']:
        if 'box2d' not in obj.keys():
            continue
        # consider only object detection task

        bbox = obj['box2d']

        label = obj['category']
        if label not in label_map:
            continue

        ymin = float(bbox['y1'])
        xmin = float(bbox['x1'])
        ymax = float(bbox['y2'])
        xmax = float(bbox['x2'])

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])

    return {'boxes': boxes, 'labels': labels}


def create_data_lists(bdd_path, output_folder):
    bdd_path = os.path.abspath(bdd_path)

    train_images = list()
    train_objects = list()
    n_objects = 0

    # Training data
    # for path in [bdd_path]:
    #     annotation_path = os.path.join(path, 'labels/bdd100k_labels_images_train.json')
    #     data = load_json(annotation_path)
    #     for dat in data:
    #         # parse annotation's json file
    #         objects = parse_annotation(dat)
    #         n_objects += len(objects)
    #         train_objects.append(objects)
    #         train_images.append(os.path.join(path, 'images/100k/train', dat['name']))
    #
    # assert len(train_objects) == len(train_images)
    #
    # # Save to file
    # dump_json(os.path.join(output_folder, 'TRAIN_images.json'), train_images)
    # dump_json(os.path.join(output_folder, 'TRAIN_objects.json'), train_objects)
    # dump_json(os.path.join(output_folder, 'label_map.json'), label_map)
    #
    # print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (
    #     len(train_images), n_objects, os.path.abspath(output_folder)))

    # Validation data
    test_images = list()
    test_objects = list()
    n_objects = 0

    for path in [bdd_path]:
        annotation_path = os.path.join(path, 'labels/bdd100k_labels_images_val.json')
        data = load_json(annotation_path)
        for dat in data:
            # parse annotation's json file
            objects = parse_annotation(dat)
            if len(objects['boxes']) == 0:
                continue
            n_objects += len(objects['boxes'])
            test_objects.append(objects)
            test_images.append(os.path.join(path, 'images/100k/val', dat['name']))

    assert len(train_objects) == len(train_images)

    dump_json(os.path.join(output_folder, 'TEST_images.json'), test_images)
    dump_json(os.path.join(output_folder, 'TEST_objects.json'), test_objects)

    print('\nThere are %d validation images containing a total of %d objects. Files have been saved to %s.' % (
        len(test_images), n_objects, os.path.abspath(output_folder)))

# bdd_path = '../assets/BDD'
# create_data_lists(bdd_path, bdd_path)