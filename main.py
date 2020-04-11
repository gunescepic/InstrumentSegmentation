import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.backends.cudnn
import cv2
from albumentations.torch.functional import img_to_tensor
import json
from datetime import datetime
from pathlib import Path
from torch.nn import functional as F

import random
import torch
import tqdm
import sys

from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Normalize,
    Compose,
    PadIfNeeded,
    RandomCrop,
    CenterCrop
)

def img_transform(p=1):
    return Compose([
        Normalize(p=1)
    ], p=p)

def cuda(x):
   	return x.cuda() if torch.cuda.is_available() else x
### VALIDATION #####

def validation_binary(model, criterion, valid_loader, num_classes=None):
    with torch.no_grad():
        model.eval()
        losses = []

        jaccard = []

        for inputs, targets in valid_loader:
            inputs = utils.cuda(inputs)
            targets = utils.cuda(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            jaccard += get_jaccard(targets, (outputs > 0).float())

        valid_loss = np.mean(losses)  # type: float

        valid_jaccard = np.mean(jaccard).astype(np.float64)

        print('Valid loss: {:.5f}, jaccard: {:.5f}'.format(valid_loss, valid_jaccard))
        metrics = {'valid_loss': valid_loss, 'jaccard_loss': valid_jaccard}
        return metrics


def get_jaccard(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    union = y_true.sum(dim=-2).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1)

    return list(((intersection + epsilon) / (union - intersection + epsilon)).data.cpu().numpy())


def validation_multi(model: nn.Module, criterion, valid_loader, num_classes):
    with torch.no_grad():
        model.eval()
        losses = []
        confusion_matrix = np.zeros(
            (num_classes, num_classes), dtype=np.uint32)
        for inputs, targets in valid_loader:
            inputs = utils.cuda(inputs)
            targets = utils.cuda(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            output_classes = outputs.data.cpu().numpy().argmax(axis=1)
            target_classes = targets.data.cpu().numpy()
            confusion_matrix += calculate_confusion_matrix_from_arrays(
                output_classes, target_classes, num_classes)

        confusion_matrix = confusion_matrix[1:, 1:]  # exclude background
        valid_loss = np.mean(losses)  # type: float
        ious = {'iou_{}'.format(cls + 1): iou
                for cls, iou in enumerate(calculate_iou(confusion_matrix))}

        dices = {'dice_{}'.format(cls + 1): dice
                 for cls, dice in enumerate(calculate_dice(confusion_matrix))}

        average_iou = np.mean(list(ious.values()))
        average_dices = np.mean(list(dices.values()))

        print(
            'Valid loss: {:.4f}, average IoU: {:.4f}, average Dice: {:.4f}'.format(valid_loss,
                                                                                   average_iou,
                                                                                   average_dices))
        metrics = {'valid_loss': valid_loss, 'iou': average_iou}
        metrics.update(ious)
        metrics.update(dices)
        return metrics


def calculate_confusion_matrix_from_arrays(prediction, ground_truth, nr_labels):
    replace_indices = np.vstack((
        ground_truth.flatten(),
        prediction.flatten())
    ).T
    confusion_matrix, _ = np.histogramdd(
        replace_indices,
        bins=(nr_labels, nr_labels),
        range=[(0, nr_labels), (0, nr_labels)]
    )
    confusion_matrix = confusion_matrix.astype(np.uint32)
    return confusion_matrix


def calculate_iou(confusion_matrix):
    ious = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = true_positives + false_positives + false_negatives
        if denom == 0:
            iou = 0
        else:
            iou = float(true_positives) / denom
        ious.append(iou)
    return ious


def calculate_dice(confusion_matrix):
    dices = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = 2 * true_positives + false_positives + false_negatives
        if denom == 0:
            dice = 0
        else:
            dice = 2 * float(true_positives) / denom
        dices.append(dice)
    return dices

moddel_list = {'UNet16': UNet16}

### INITIALIZE VARIABLES FOR TRAINING ###
###									  ###
### We have 16 videos in all, so 4 fold cross validation will be applied and arrange it. ####

device_ids = '0,1,2'
lr =  0.0001
batch_size = 6
n_epochs = 10
jaccard_weight = 0.3
train_crop_height = 1024
train_crop_width = 1280
val_crop_height = 1024 
val_crop_width = 1024 
fold = 4
segmentation_type = 'binary'

data_path = Path('data')

train_path = data_path / 'train'

cropped_train_path = data_path / 'cropped_train' # new path for cropped images

original_height, original_width = 1080, 1920
height, width = 1024, 1280
h_start, w_start = 28, 320

binary_factor = 255
parts_factor = 85
instrument_factor = 32

worker = 12

if segmentation_type == 'parts':
    num_classes = 4 #not sure
elif segmentation_type == 'instruments':
    num_classes = 16
else:
    num_classes = 1


root = 'runs/debug'
root.mkdir(exist_ok=True, parents=True)
    

############# LOAD DATA ################################
# import torchvision.transforms.functional as F


# def img_to_tensor(im, normalize=None):
#     tensor = torch.from_numpy(np.moveaxis(im / (255. if im.dtype == np.uint8 else 1), -1, 0).astype(np.float32))
#     if normalize is not None:
#         return F.normalize(tensor, **normalize)
#     return tensor


class RoboticsDataset(Dataset):
    def __init__(self, file_names, to_augment=False, transform=None, mode='train', problem_type=None):
        self.file_names = file_names
        self.to_augment = to_augment
        self.transform = transform
        self.mode = mode
        self.problem_type = problem_type

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        image = load_image(img_file_name)
        mask = load_mask(img_file_name, segmentation_type)

        data = {"image": image, "mask": mask}
        augmented = self.transform(**data)
        image, mask = augmented["image"], augmented["mask"]

        if self.mode == 'train':
            if self.problem_type == 'binary':
                return img_to_tensor(image), torch.from_numpy(np.expand_dims(mask, 0)).float()
            else:
                return img_to_tensor(image), torch.from_numpy(mask).long()
        else:
            return img_to_tensor(image), str(img_file_name)


def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path, problem_type):
    if problem_type == 'binary':
        mask_folder = 'binary_masks'
        factor = binary_factor
    elif problem_type == 'parts':
        mask_folder = 'parts_masks'
        factor = parts_factor
    elif problem_type == 'instruments':
        factor = instrument_factor
        mask_folder = 'instruments_masks'

    mask = cv2.imread(str(path).replace('images', mask_folder).replace('jpg', 'png'), 0)

    return (mask / factor).astype(np.uint8)

############# PREPROCESS DATA ##########################
"""
[1] Merge masks with different instruments into one binary mask
[2] Crop black borders from images and masks
"""


for instrument_index in range(1, 17):
    instrument_folder = 'seq_' + str(instrument_index)

    (cropped_train_path / instrument_folder / 'left_frames').mkdir(exist_ok=True, parents=True)

    binary_mask_folder = (cropped_train_path / instrument_folder / 'binary_masks')
    binary_mask_folder.mkdir(exist_ok=True, parents=True)

    parts_mask_folder = (cropped_train_path / instrument_folder / 'parts_masks')
    parts_mask_folder.mkdir(exist_ok=True, parents=True)

    instrument_mask_folder = (cropped_train_path / instrument_folder / 'instruments_masks')
    instrument_mask_folder.mkdir(exist_ok=True, parents=True)

    mask_folders = list((train_path / instrument_folder / labels).glob('*'))
    # mask_folders = [x for x in mask_folders if 'Other' not in str(mask_folders)]

    for file_name in tqdm(list((train_path / instrument_folder / 'left_frames').glob('*'))):
        img = cv2.imread(str(file_name))
        old_h, old_w, _ = img.shape

        img = img[h_start: h_start + height, w_start: w_start + width]
        cv2.imwrite(str(cropped_train_path / instrument_folder / 'left_frames' / (file_name.stem + '.png')), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 100])

        mask_binary = np.zeros((old_h, old_w))
        mask_parts = np.zeros((old_h, old_w))
        mask_instruments = np.zeros((old_h, old_w))

        for mask_folder in mask_folders:
            mask = cv2.imread(str(mask_folder / file_name.name), 0)

            if 'Bipolar_Forceps' in str(mask_folder):
                mask_instruments[mask > 0] = 1
            elif 'Prograsp_Forceps' in str(mask_folder):
                mask_instruments[mask > 0] = 2
            elif 'Large_Needle_Driver' in str(mask_folder):
                mask_instruments[mask > 0] = 3
            elif 'Vessel_Sealer' in str(mask_folder):
                mask_instruments[mask > 0] = 4
            elif 'Grasping_Retractor' in str(mask_folder):
                mask_instruments[mask > 0] = 5
            elif 'Monopolar_Curved_Scissors' in str(mask_folder):
                mask_instruments[mask > 0] = 6
            elif 'Other' in str(mask_folder):
                mask_instruments[mask > 0] = 7

            if 'Other' not in str(mask_folder):
                mask_binary += mask

                mask_parts[mask == 10] = 1  # Shaft
                mask_parts[mask == 20] = 2  # Wrist
                mask_parts[mask == 30] = 3  # Claspers

        mask_binary = (mask_binary[h_start: h_start + height, w_start: w_start + width] > 0).astype(
            np.uint8) * binary_factor
        mask_parts = (mask_parts[h_start: h_start + height, w_start: w_start + width]).astype(
            np.uint8) * parts_factor
        mask_instruments = (mask_instruments[h_start: h_start + height, w_start: w_start + width]).astype(
            np.uint8) * instrument_factor

        cv2.imwrite(str(binary_mask_folder / file_name.name), mask_binary)
        cv2.imwrite(str(parts_mask_folder / file_name.name), mask_parts)
        cv2.imwrite(str(instrument_mask_folder / file_name.name), mask_instruments)

def get_split(fold):

	#4 fold cross validation can be changed

    folds = {0: [1, 5, 9, 13],
             1: [2, 6, 10, 14],
             2: [3, 7, 11, 15],
             3: [4, 8, 12, 16]}

    train_path = data_path / 'cropped_train'

    train_file_names = []
    val_file_names = []

    for instrument_id in range(1, 9):
        if instrument_id in folds[fold]:
            val_file_names += list((train_path / ('seq_' + str(instrument_id)) / 'left_frames').glob('*'))
        else:
            train_file_names += list((train_path / ('seq_' + str(instrument_id)) / 'left_frames').glob('*'))

    return train_file_names, val_file_names

############# CREATE MODEL ############################
from torchvision import models
import torchvision


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)

### Unet with encoder pretrained vgg16 ###

class UNet16(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG11
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu,
                                   self.encoder[2],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   self.relu,
                                   self.encoder[7],
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   self.relu,
                                   self.encoder[12],
                                   self.relu,
                                   self.encoder[14],
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],
                                   self.relu,
                                   self.encoder[19],
                                   self.relu,
                                   self.encoder[21],
                                   self.relu)

        self.conv5 = nn.Sequential(self.encoder[24],
                                   self.relu,
                                   self.encoder[26],
                                   self.relu,
                                   self.encoder[28],
                                   self.relu)

        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8)

        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = self.final(dec1)

        return x_out


############# TRAIN MODEL ##############################
## ADD LOSS ####
class LossBinary:
"""
    Loss defined as \alpha BCE - (1 - \alpha) SoftJaccard
    """

    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1).float()
            jaccard_output = F.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss


class LossMulti:
    def __init__(self, jaccard_weight=0, class_weights=None, num_classes=1):
        if class_weights is not None:
            nll_weight = utils.cuda(
                torch.from_numpy(class_weights.astype(np.float32)))
        else:
            nll_weight = None
        self.nll_loss = nn.NLLLoss2d(weight=nll_weight)
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            for cls in range(self.num_classes):
                jaccard_target = (targets == cls).float()
                jaccard_output = outputs[:, cls].exp()
                intersection = (jaccard_output * jaccard_target).sum()

                union = jaccard_output.sum() + jaccard_target.sum()
                loss -= torch.log((intersection + eps) / (union - intersection + eps)) * self.jaccard_weight
        return loss

model = UNet16(num_classes=num_classes, pretrained=True) 


# cuda dependency check later

if torch.cuda.is_available():
    if device_ids:
        device_ids = list(map(int, device_ids.split(',')))
    else:
        device_ids = None
    model = nn.DataParallel(model, device_ids=device_ids).cuda()
else:
    raise SystemError('GPU device not found')
if segmentation_type == 'binary':
    loss = LossBinary(jaccard_weight=jaccard_weight)
else:
    loss = LossMulti(num_classes=num_classes, jaccard_weight=jaccard_weight)

cudnn.benchmark = True

train_file_names, val_file_names = get_split(fold)

print('num train = {}, num_val = {}'.format(len(train_file_names), len(val_file_names)))

def train_transform(p=1):
    return Compose([
        PadIfNeeded(min_height=train_crop_height, min_width=train_crop_width, p=1),
        RandomCrop(height=train_crop_height, width=train_crop_width, p=1),
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        Normalize(p=1)
    ], p=p)

def val_transform(p=1):
    return Compose([
        PadIfNeeded(min_height=val_crop_height, min_width=val_crop_width, p=1),
        CenterCrop(height=val_crop_height, width=val_crop_width, p=1),
        Normalize(p=1)
    ], p=p)

train_loader = DataLoader(
	dataset = RoboticsDataset(train_file_names, transform = train_transform(p = 1), problem_type = segmentation_type),
	shuffle = True,
	num_workers = workers,
	batch_size = batch_size,
	pin_memory = torch.cuda.is_available()
	)
valid_loader = DataLoader(
	dataset = RoboticsDataset(val_file_names, transform = val_transform(p = 1), problem_type = segmentation_type),
	shuffle = True,
	num_workers = workers,
	batch_size = batch_size
	)
# root.joinpath('params.json').write_text(
#         json.dumps(vars(args), indent=True, sort_keys=True))

if segmentation_type == 'binary':
    valid = validation_binary
else:
    valid = validation_multi




def write_event(log, step, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def check_crop_size(image_height, image_width):
    """Checks if image size divisible by 32.

    Args:
        image_height:
        image_width:

    Returns:
        True if both height and width divisible by 32 and False otherwise.

    """
    return image_height % 32 == 0 and image_width % 32 == 0


def train(args, model, criterion, train_loader, valid_loader, validation, init_optimizer, n_epochs=None, fold=None,
          num_classes=None):
    lr = lr
    n_epochs = n_epochs
    optimizer = init_optimizer(lr)

    root = Path(root)
    model_path = root / 'model_{fold}.pt'.format(fold=fold)
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, str(model_path))

    report_each = 10
    log = root.joinpath('train_{fold}.log'.format(fold=fold)).open('at', encoding='utf8')
    valid_losses = []
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        random.seed()
        tq = tqdm.tqdm(total=(len(train_loader) * batch_size))
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        tl = train_loader
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                inputs = cuda(inputs)

                with torch.no_grad():
                    targets = cuda(targets)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                batch_size = inputs.size(0)
                loss.backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)
                losses.append(loss.item())
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)
            write_event(log, step, loss=mean_loss)
            tq.close()
            save(epoch + 1)
            valid_metrics = validation(model, criterion, valid_loader, num_classes)
            write_event(log, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return
train(
    init_optimizer=lambda lr: Adam(model.parameters(), lr=lr),
    args=args,
    model=model,
    criterion=loss,
    train_loader=train_loader,
    valid_loader=valid_loader,
    validation=valid,
    fold=4, #args.fold
    num_classes=num_classes
)
############# GENERATE MASKS ##########################

## CHECK THIS METHOD ###
def predict(model, from_file_names, batch_size, to_path, problem_type, img_transform):
    loader = DataLoader(
        dataset=RoboticsDataset(from_file_names, transform=img_transform, mode='predict', problem_type=problem_type),
        shuffle=False,
        batch_size=6,
        num_workers=workers,
        pin_memory=torch.cuda.is_available()
    )

    with torch.no_grad():
        for batch_num, (inputs, paths) in enumerate(tqdm(loader, desc='Predict')):
            inputs = utils.cuda(inputs)

            outputs = model(inputs)

            for i, image_name in enumerate(paths):
                if problem_type == 'binary':
                    factor = prepare_data.binary_factor
                    t_mask = (F.sigmoid(outputs[i, 0]).data.cpu().numpy() * factor).astype(np.uint8)
                elif problem_type == 'parts':
                    factor = prepare_data.parts_factor
                    t_mask = (outputs[i].data.cpu().numpy().argmax(axis=0) * factor).astype(np.uint8)
                elif problem_type == 'instruments':
                    factor = prepare_data.instrument_factor
                    t_mask = (outputs[i].data.cpu().numpy().argmax(axis=0) * factor).astype(np.uint8)

                h, w = t_mask.shape

                full_mask = np.zeros((original_height, original_width))
                full_mask[h_start:h_start + h, w_start:w_start + w] = t_mask

                instrument_folder = Path(paths[i]).parent.parent.name

                (to_path / instrument_folder).mkdir(exist_ok=True, parents=True)

                cv2.imwrite(str(to_path / instrument_folder / (Path(paths[i]).stem + '.png')), full_mask)

model = UNet16(num_classes = num_classes)
### needs path to saved trained model above ###
state = torch.load(str(Path(model_path)).joinpath('model_{fold}.pt'.format(fold=args.fold)))
state = {key.replace('module.', ''): value for key, value in state['model'].items()}
model.load_state_dict(state)

if torch.cuda.is_available():
    return model.cuda()

model.eval()

_, file_names = get_split(fold)
print('num file_names = {}'.format(len(file_names)))


output_path = 'predictions/unet16/binary'
output_path.mkdir(exist_ok=True, parents=True)

predict(model, file_names, batch_size, output_path, problem_type=segmentation_type,
        img_transform=img_transform(p=1))

############# EVALUATE ################################
result_dice = []
result_jaccard = []
train_path = 'predictions/unet16'
target_path = 'data/cropped_train'

if segmentation_type == 'binary':
    for instrument_id in tqdm(range(1, 9)):
        instrument_dataset_name = 'seq_' + str(instrument_id)

        for file_name in (
                train_path / instrument_dataset_name / 'binary_masks').glob('*'):
            y_true = (cv2.imread(str(file_name), 0) > 0).astype(np.uint8)

            pred_file_name = target_path / 'binary' / instrument_dataset_name / file_name.name

            pred_image = (cv2.imread(str(pred_file_name), 0) > 255 * 0.5).astype(np.uint8)
            y_pred = pred_image[h_start:h_start + height, w_start:w_start + width]

            result_dice += [dice(y_true, y_pred)]
            result_jaccard += [jaccard(y_true, y_pred)]

elif args.problem_type == 'parts':
    for instrument_id in tqdm(range(1, 9)):
        instrument_dataset_name = 'seq_' + str(instrument_id)
        for file_name in (
                train_path / instrument_dataset_name / 'parts_masks').glob('*'):
            y_true = cv2.imread(str(file_name), 0)

            pred_file_name = target_path / 'parts' / instrument_dataset_name / file_name.name

            y_pred = cv2.imread(str(pred_file_name), 0)[h_start:h_start + height, w_start:w_start + width]

            result_dice += [general_dice(y_true, y_pred)]
            result_jaccard += [general_jaccard(y_true, y_pred)]

elif args.problem_type == 'instruments':
    for instrument_id in tqdm(range(1, 9)):
        instrument_dataset_name = 'seq_' + str(instrument_id)
        for file_name in (
                train_path / instrument_dataset_name / 'instruments_masks').glob('*'):
            y_true = cv2.imread(str(file_name), 0)

            pred_file_name = target_path / 'instruments' / instrument_dataset_name / file_name.name

            y_pred = cv2.imread(str(pred_file_name), 0)[h_start:h_start + height, w_start:w_start + width]

            result_dice += [general_dice(y_true, y_pred)]
            result_jaccard += [general_jaccard(y_true, y_pred)]

print('Dice = ', np.mean(result_dice), np.std(result_dice))
print('Jaccard = ', np.mean(result_jaccard), np.std(result_jaccard))
