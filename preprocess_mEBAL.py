import os
import numpy as np

root = '/db/mEBAL/Eye Blinks'
out = '/db/mEBAL/data.txt'
train_out = '/db/mEBAL/traindata.txt'
test_out = '/db/mEBAL/testdata.txt'

file1 = '/0.png'
file2 = '/1.png'
file3 = '/8.png'
file4 = '/9.png'
file5 = '/10.png'
file6 = '/17.png'
file7 = '/18.png'

sep = '$'

train_size = 0.95

file = open(out, 'w+')
trf = open(train_out, 'w+')
tef = open(test_out, 'w+')

users = os.listdir(root)
for user in users:
    user_path = os.path.join(root, user, 'left eye')
    aux_user_path = os.path.join(root, user, 'right eye')

    blinks = os.listdir(user_path)

    train_part = int(train_size * len(blinks))
    train_blinks = np.random.choice(blinks, size=train_part)

    for blink in blinks:
        blink_path = os.path.join(user_path, blink)
        aux_path = os.path.join(aux_user_path, blink)

        images = os.listdir(blink_path)
        num_images = len(images)
        if num_images == 19:
            file.write(f'{blink_path}{file1}{sep}{aux_path}{file1}{sep}0\n')
            file.write(f'{blink_path}{file2}{sep}{aux_path}{file2}{sep}0\n')
            file.write(f'{blink_path}{file3}{sep}{aux_path}{file3}{sep}1\n')
            file.write(f'{blink_path}{file4}{sep}{aux_path}{file4}{sep}1\n')
            file.write(f'{blink_path}{file5}{sep}{aux_path}{file5}{sep}1\n')
            file.write(f'{blink_path}{file6}{sep}{aux_path}{file6}{sep}0\n')
            file.write(f'{blink_path}{file7}{sep}{aux_path}{file7}{sep}0\n')

            if blink in train_blinks:
                trf.write(f'{blink_path}{file1}{sep}{aux_path}{file1}{sep}0\n')
                trf.write(f'{blink_path}{file2}{sep}{aux_path}{file2}{sep}0\n')
                trf.write(f'{blink_path}{file3}{sep}{aux_path}{file3}{sep}1\n')
                trf.write(f'{blink_path}{file4}{sep}{aux_path}{file4}{sep}1\n')
                trf.write(f'{blink_path}{file5}{sep}{aux_path}{file5}{sep}1\n')
                trf.write(f'{blink_path}{file6}{sep}{aux_path}{file6}{sep}0\n')
                trf.write(f'{blink_path}{file7}{sep}{aux_path}{file7}{sep}0\n')
            else:
                tef.write(f'{blink_path}{file1}{sep}{aux_path}{file1}{sep}0\n')
                tef.write(f'{blink_path}{file2}{sep}{aux_path}{file2}{sep}0\n')
                tef.write(f'{blink_path}{file3}{sep}{aux_path}{file3}{sep}1\n')
                tef.write(f'{blink_path}{file4}{sep}{aux_path}{file4}{sep}1\n')
                tef.write(f'{blink_path}{file5}{sep}{aux_path}{file5}{sep}1\n')
                tef.write(f'{blink_path}{file6}{sep}{aux_path}{file6}{sep}0\n')
                tef.write(f'{blink_path}{file7}{sep}{aux_path}{file7}{sep}0\n')

file.close()
trf.close()
tef.close()
