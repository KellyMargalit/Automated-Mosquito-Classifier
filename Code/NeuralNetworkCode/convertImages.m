srcFiles = dir('PupaeRotatedTest/Male/*.jpeg');  % the folder in which ur images exists

for i = 1 : length(srcFiles)
    filename = strcat('PupaeRotatedTest/Male/',srcFiles(i).name);
    I = imread(filename);
    I2 = rgb2gray(I);
    I3 = cat(3, I2, I2, I2);
    imwrite(I3,fullfile('GrayRGBTest/Male/', srcFiles(i).name));
end