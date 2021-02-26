
def to_yolo(learning_loop_box, image_width, image_height, categories):
    w = float(learning_loop_box['width']) / float(image_width)
    h = float(learning_loop_box['height']) / float(image_height)
    x = (float((learning_loop_box['x']) + float(learning_loop_box['width']) / 2) / float(image_width))
    y = (float((learning_loop_box['y']) + float(learning_loop_box['height']) / 2) / float(image_height))

    yoloID = categories.index(learning_loop_box['category_id'])

    return ' '.join([
        str(yoloID),
        str("%.6f" % x),
        str("%.6f" % y),
        str("%.6f" % w),
        str("%.6f" % h)])


def create_data_file(trainings_folder: str, number_of_classes: int) -> None:
    number_of_classes = f'classes = {number_of_classes}'
    train = 'train  = train.txt'
    valid = 'valid  = test.txt'
    names = 'names = names.txt'
    backup = 'backup = backup/'
    with open(f'{trainings_folder}/data.txt', 'w') as f:
        data_object = [number_of_classes, train, valid, names, backup]
        f.write('\n'.join(data_object))
