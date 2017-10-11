% img = imread('/home/yandong/eclipse-workspace/cv2_quiz1/Images/susan_input1.png');
img = im2double(imread('../Images/susan_input1.png'));
[map r c] = susanCorner(img);
figure,imshow(img),hold on
plot(c,r,'o')