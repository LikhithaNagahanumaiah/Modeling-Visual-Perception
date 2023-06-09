%% Programming Assignments 
%Method 2
clc;
clear all;
close all;
%Generating 1920 pixels evenly spaced between 400-700nm 
plen = linspace(400,700,1920);
%% For 1nm
a = zeros(301,301); %generating sqaure matrix of zeros,n*m=301*301.
w1nm = 400:1:700; %spectrum with 1nm step
%o1nm= tri(w1nm)
for i = 1:1:301 %for loop to create a triangle and replacing them inside matrix 'a'
  a(i,:) = triangularPulse(w1nm(i)-5, w1nm(i), w1nm(i)+5,w1nm); %crreating a triangle
end
out1nm = interp1(w1nm, a, linspace(400,700,1920)); %interpolation of 1-D func using linear interpolation
%new= repmat(1080,out1nm);
figure;
plot(plen,out1nm(:,1:1:100));%plotting spectral power at each pixel for all pixels at interval of 1
title('1nm: Plotting pixel values from (1,1)-(100,100)in interval of 1')
xlabel('wavelength(nm)');
ylabel('relative spectral power');
figure;
plot(plen,out1nm(:,1:38:301));%plotting spectral power at pixel position with interval of 38
title('1nm: Plotting pixel values from(1,1)-(301,301)in interval of 38')
xlabel('wavelength(nm)');
ylabel('relative spectral power')

% 2-D to multidimention arrays 

%% For 5nm
b = zeros(61,61);% there are 61 wavelegths when step is 5nm
w5nm = 400:5:700;%spectrum with 5nm step
for i = 1:1:61 %for loop to create a triangle and replacing them inside matrix 'b'
  b(i,:) = triangularPulse(w5nm(i)-5, w5nm(i), w5nm(i)+5,w5nm);
end%for
out5nm = interp1(w5nm, b, linspace(400,700,1920));
figure;
plot(plen,out5nm(:,1:1:61));%plotting spectral power at each pixel from (1,1)-(61,61) for each pixel
title('5nm: Plotting pixel values from(1,1)-(61,61)in interval of 1')
xlabel('wavelength(nm)');
ylabel('relative spectral power');
figure;
plot(plen,out5nm(:,1:10:50));%plotting spectral power at pixel value from (1,1)-(50,50) interval of 10
title('1nm: Plotting pixel values from(1,1)-(50,50)in interval of 10')
title('Plotting spectral power at 10 interval for 50 wavelength')
xlabel('wavelength(nm)');
ylabel('relative spectral power')


%% Method-1
wl1 = 400:1:700;% wavelength with 1nm step
wl5 = 400:5:700;% wavelength with 5nm step
o1nm = dimension(1920,1080,1);%sendng value to function

pxl1 = o1nm(1,1,:);%accessing(1,1) pixel 
pxl1 = reshape(pxl1,1,[]);% converting to 2-d from 3-d
pxl2 = o1nm(200,300,:);
pxl2 = reshape(pxl2,1,[]);
pxl3 = o1nm(1010,400,:);
pxl3 = reshape(pxl3,1,[]);

figure %plot the triangle bandpass at mentioned pixel value above
plot(wl1,pxl1);
hold on 
plot(wl1,pxl2);
hold on
plot(wl1,pxl3);
title('Plot 1nm:pixels plot at(1,1),(200,300),(1010,400)')
xlabel('wavelength(nm)');
ylabel('relative spectral power');

o5nm = dimension(1920,1080,5); % 5nm processing
pxl4 = o5nm(1,1,:);
pxl4 = reshape(pxl4,1,[]);% converting to 2-d from 3-d
pxl5 = o5nm(300,40,:);
pxl5 = reshape(pxl5,1,[]);
pxl6 = o5nm(500,1000,:);
pxl6 = reshape(pxl6,1,[]);
figure %plot the triangle bandpass
plot(wl5,pxl4);
hold on 
plot(wl5,pxl5);
hold on
plot(wl5,pxl6);
title('Plot 5nm:pixels plot at(1,1),(300,40),(500,1000)')
xlabel('wavelength(nm)');
ylabel('relative spectral power');
% figure;%plot the triangle bandpass
% title('plot at')
% xlabel('wavelength(nm)');
% ylabel('relative spectral power');
% plot(wl5,p4);
% figure;
% title('plot at ')
% xlabel('wavelength(nm)');
% ylabel('relative spectral power'); 
% plot(wl5,p5);
% figure;
% title('plot at ')
% xlabel('wavelength(nm)');
% ylabel('relative spectral power');
% plot(wl5,p6);

%% 1nm image
dlambda1 = 1;%d lamba 1nm
funcn1 = (readtable('CIE_all_1nm_data.xlsx'));
xb1 = str2double(table2array(funcn1(104:1:404,6)));%xbar from 400-700nm 
yb1 = str2double(table2array(funcn1(104:1:404,7)));%ybar from 400-700nm 
zb1 = str2double(table2array(funcn1(104:1:404,8)));%zbar from 400-700nm 
std_ill1= readtable('Illuminant Data.xlsx');

pixel1 = squeeze(o1nm(1,:,:));%reducing dimentionality of 1st row

%k= 100/(S_lambda'* yb*dlambda1);
X1=  (pixel1* xb1*dlambda1); %get XYZ 
Y1= (pixel1* yb1*dlambda1);
Z1= (pixel1* zb1 *dlambda1); 
xyz1= [X1 Y1 Z1]; 

RGB1 =(xyz2rgb(xyz1/10)); %to get value in correct range
%rgb1 = lin2rgb(RGB1);
length= 1080;
width1 = 1920;
image1 = reshape(ones(length,1)*reshape(RGB1,1,width1*3),[length,width1,3]);% replicate 1080
% img= repmat(rgb,[2 1]);
%hsv= rgb2hsv(RGB);
for i = 1:3
    image1(:,i)=(image1(:,i)+abs(min(image1(:,i)))) / max(image1(:,i)+abs(min(image1(:,i))));%normalize
end
%rgbimg= min(max(image,0),1);
figure;
outimg= imshow(image1);

%% Question 8
%Human vision
x2lm= xyz2lms(xyz1/10,3,'linear');
lm2x= lms2xyz(x2lm);
fin= xyz2rgb((lm2x)); 
length= 1080;
width1 = 1920;
img_tri = reshape(ones(length,1)*reshape(fin,1,width1*3),[length,width1,3]);% replicate 1080
for i = 1:3
    img_tri(:,i)=(img_tri(:,i)+abs(min(img_tri(:,i)))) / max(img_tri(:,i)+abs(min(img_tri(:,i))));%normalize
end
figure,imshow(img_tri);
title('Tritanope Vision- missing S cone')
%Protanopia observer, missing L cones
x2l= xyz2lms(xyz1/10,2,'linear');
l2x= lms2xyz(x2l);
fin_l= xyz2rgb((l2x)); 
length= 1080;
width1 = 1920;
img_pro = reshape(ones(length,1)*reshape(fin_l,1,width1*3),[length,width1,3]);% replicate 1080
for i = 1:3
    img_pro(:,i)=(img_pro(:,i)+abs(min(img_pro(:,i)))) / max(img_pro(:,i)+abs(min(img_pro(:,i))));%normalize
end
figure,imshow(img_pro);
title('Deuteranope Vision-Missing M cones')
%% Animal vision

r_uv=table2array(readtable('Uv.xlsx'));
uv=interp1(r_uv(:,1),r_uv(:,2),w5nm');
uv(isnan(uv))=0;
uc=repmat(reshape(uv,1,1,75),1080,1920,1);
img2duv=sum(o5nm.*uc,3);
figure, imshow(img2duv,[])
title('UV-grayimage')

r_l=table2array(readtable('L_cones.xlsx'));
lcone=interp1(r_l(:,1),r_l(:,2),w5nm');
lcone(isnan(lcone))=0;
lc=repmat(reshape(lcone,1,1,75),1080,1920,1);
img2dl=sum(o5nm.*lc,3);
figure, imshow(img2dl,[])
title('L-grayimage')


r_m=table2array(readtable('M_cones.xlsx'));
mcone=interp1(r_m(:,1),r_m(:,2),w5nm');
mcone(isnan(mcone))=0;
mc=repmat(reshape(mcone,1,1,75),1080,1920,1);
img2dm=sum(o5nm.*mc,3);
figure, imshow(img2dm,[])
title('M-grayimage')


r_s=table2array(readtable('S_cone.xlsx'));
scone=interp1(r_s(:,1),r_s(:,2),w5nm');
scone(isnan(scone))=0;
sc=repmat(reshape(scone,1,1,75),1080,1920,1);
img2ds=sum(o5nm.*mc,3);
figure, imshow(img2ds,[])
title('S-grayimage')

full_img1= cat(3,img2duv,img2dl,img2dm);
figure, imshow(full_img1)
title('Animal UV,L,M cones')

full_img= cat(3,img2dl,img2dm,img2ds);
figure, imshow(full_img)
title('Animal LMS')

%% Question 6,7
%xyz1;
wp_d65 = whitepoint('d65');
appearance1 = CAM02(xyz1,wp_d65,'D',1,'adaptingLuminance',2,'whiteLuminance',10);
xyz_1 = inverse_CAM02(appearance1,wp_d65);
rgb_1= xyz2rgb(xyz_1);
appearance2 = CAM02(xyz1,wp_d65,'D',1,'adaptingLuminance',20,'whiteLuminance',100);
xyz_2 = inverse_CAM02(appearance2,wp_d65);
rgb_2= xyz2rgb(xyz_2);
appearance3 = CAM02(xyz1,wp_d65,'D',1,'adaptingLuminance',2000,'whiteLuminance',10000);
xyz_3 = inverse_CAM02(appearance3,wp_d65);
rgb_3= xyz2rgb(xyz_3);
rgb_new=[rgb_1 rgb_2 rgb_3];
rgb_new= rgb_new/max(max((rgb_new)));
length= 1080;
width1 = 1920;
rgb_re = reshape(ones(length,1)*reshape(rgb_new,1,width1*9),[length,width1,9]);

% figure
% imshow(rgb_re(:,:,1:3))
% figure
% imshow(rgb_re(:,:,4:6))
% figure
% imshow(rgb_re(:,:,7:9))

%% Question 6 - lightness and brightness 
jrgb1= repmat(appearance1.J,1,3);
jrgb2= repmat(appearance2.J,1,3);
jrgb3= repmat(appearance3.J,1,3);
jrgb=[jrgb1 jrgb2 jrgb3];
jrgb= jrgb/max(max((jrgb)));
length= 1080;
width1 = 1920;
jrgbf = reshape(ones(length,1)*reshape(jrgb,1,width1*9),[length,width1,9]);

figure
imshow(jrgbf(:,:,1:3))
title('Lightness, Lum=10')
figure
imshow(jrgbf(:,:,4:6))
title('Lightness, Lum=100')
figure
imshow(jrgbf(:,:,7:9))
title('Lightness, Lum=10000')
%Brightness
qrgb1= repmat(appearance1.Q,1,3);
qrgb2= repmat(appearance2.Q,1,3);
qrgb3= repmat(appearance3.Q,1,3);
qrgb=[qrgb1 qrgb2 qrgb3];
qrgb= qrgb/max(max((qrgb)));
length= 1080;
width1 = 1920;
qrgbf = reshape(ones(length,1)*reshape(qrgb,1,width1*9),[length,width1,9]);

figure
imshow(qrgbf(:,:,1:3))
title('Brightness, Lum=10')
figure
imshow(qrgbf(:,:,4:6))
title('Brightness, Lum=100')
figure
imshow(qrgbf(:,:,7:9))
title('Brightness, Lum=10000')
%% 
Y= xyz1(:,2);
length= 1080;
width1 = 1920;
rep_y= repmat(Y,1,3);
yy= rep_y/max(max((rep_y)));
lum_Y = reshape(ones(length,1)*reshape(yy,1,width1*3),[length,width1,3]);
figure
imshow(lum_Y)
title('Luminance-of image')
%% Question 7 Colorfulness and saturation
mrgb1= repmat(appearance1.M,1,3);
mrgb2= repmat(appearance2.M,1,3);
mrgb3= repmat(appearance3.M,1,3);
mrgb=[mrgb1 mrgb2 mrgb3];
mrgb= mrgb/max(max((mrgb)));
length= 1080;
width1 = 1920;
mrgbf = reshape(ones(length,1)*reshape(mrgb,1,width1*9),[length,width1,9]);

figure
imshow(mrgbf(:,:,1:3))
title('Colorfulness, Lum=10')
figure
imshow(mrgbf(:,:,4:6))
title('Colorfulness, Lum=100')
figure
imshow(mrgbf(:,:,7:9))
title('Colorfulness, Lum=10000')
%saturation
srgb1= repmat(appearance1.s,1,3);
srgb2= repmat(appearance2.s,1,3);
srgb3= repmat(appearance3.s,1,3);
srgb=[srgb1 srgb2 srgb3];
srgb= srgb/max(max((srgb)));
length= 1080;
width1 = 1920;
srgbf = reshape(ones(length,1)*reshape(srgb,1,width1*9),[length,width1,9]);

figure
imshow(srgbf(:,:,1:3))
title('saturation, Lum=10')
figure
imshow(srgbf(:,:,4:6))
title('saturation, Lum=100')
figure
imshow(srgbf(:,:,7:9))
title('saturation, Lum=10000')
%Brightness,colorfullness and hue
ap1={};
ap1{1}.Q= appearance1.Q;
ap1{1}.M= appearance1.M;
ap1{1}.h= appearance1.h;
ap2={};
ap2{1}.Q= appearance2.Q;
ap2{1}.M= appearance2.M;
ap2{1}.h= appearance2.h;
ap3={};
ap3{1}.Q= appearance3.Q;
ap3{1}.M= appearance3.M;
ap3{1}.h= appearance3.h;
colr_app1 = inverse_CAM02(ap1{1},wp_d65,'D',1,'adaptingLuminance',2,'whiteLuminance',10);
clr1= xyz2rgb(colr_app1);
colr_app2 = inverse_CAM02(ap2{1},wp_d65,'D',1,'adaptingLuminance',2,'whiteLuminance',10);
clr2= xyz2rgb(colr_app2);
colr_app3 = inverse_CAM02(ap3{1},wp_d65,'D',1,'adaptingLuminance',2,'whiteLuminance',10);
clr3= xyz2rgb(colr_app3);
clr_rgb=[clr1 clr2 clr3];
clr_rgb= clr_rgb/max(max((clr_rgb)));
length= 1080;
width1 = 1920;
clr_new = reshape(ones(length,1)*reshape(clr_rgb,1,width1*9),[length,width1,9]);

figure
imshow(clr_new(:,:,1:3))
title('Color-QMh, Lum=10')
figure
imshow(clr_new(:,:,4:6))
title('Color-QMh, Lum=100')
figure
imshow(clr_new(:,:,7:9))
title('Color-QMh, Lum=10000')
%% 5nm image

dlambda5 = 5;%d lamba 1nm
funcn = (readtable('CIE_all_1nm_data.xlsx'));
xb = str2double(table2array(funcn(104:5:404,6)));%xbar from 400-700nm 
yb = str2double(table2array(funcn(104:5:404,7)));%ybar from 400-700nm 
zb = str2double(table2array(funcn(104:5:404,8)));%zbar from 400-700nm 
%std_ill= readtable('Illuminant Data.xlsx');
%S_lambda = table2array(std_ill(21:81,5)); % D65 display 

pixel = squeeze(o5nm(1,:,:));
k=1;
%k= 100/(S_lambda'* yb*dlambda1);
X=  k*(pixel* xb*dlambda5); %reducing dimentionality of 1st row
Y= k*(pixel* yb*dlambda5);
Z= k*(pixel* zb *dlambda5); 

xyz= [X Y Z]; %XYZ value
RGB =(xyz2rgb(xyz));
%rgb = lin2rgb(RGB);
length= 1080;
width = 1920;
image = reshape(ones(length,1)*reshape(RGB,1,width*3),[length,width,3]);%replicate 1080
% img= repmat(rgb,[2 1]);
%hsv= rgb2hsv(RGB);
for i = 1:3
    image(:,i)=(image(:,i)+abs(min(image(:,i)))) / max(image(:,i)+abs(min(image(:,i)))); %normalize
end
%rgbimg= min(max(image,0),1);
figure;
imshow(image)

%% H-W-3 create a box of 1080*1920 
figure;
plot([1920 1920],[0,1080],'k-');
hold on;
plot([0 1920],[1080 1080],'k-');
title('Outlining of 1920x1080')
%adjust radius and the number of circle in each row to find the best arrangement
r = 71; %radius considered as 71 
cone_rw = 13; % number of rows , %104 cone aperture = 13*8 
%l:m:s = 8:4:1;
%randomsample returns weighted sample, sampled for 104 value    
smpl = randsample([1 2 3],104,true,[0.4/0.65 0.2/0.65 0.05/0.65]); 
for i = 1:104
  c_h = fix((i+cone_rw-1)/cone_rw); %round to zero
  c_w = rem(i+cone_rw-1,cone_rw)+1; %remainder 
  ih = ceil(sqrt(3)*(c_h-1)*r+r)+40; 
  if rem(c_h,2)
    iw = r+(c_w-1)*2*r; %finding circle from center
  else
    iw = 2*r+(c_w-1)*2*r;
  end
  one_cone(i,:) = [ih,iw,smpl(i)]; %complete one cone
  angl = 0 : (2 * pi / 10000) : (2 * pi);
  lin_x = r * cos(angl) + iw;
  lin_y = r * sin(angl) + ih;
  plot(lin_x, lin_y, '-');
  hold on;
end
hold off;
[x,y]=meshgrid(1:1920,1:1080); %rectangular grid in 2d
cone_img = zeros(1080, 1920); 
for i = 1:104
cone_img((x - one_cone(i,2)).^2 + (y - one_cone(i,1)).^2 <= r.^2) = one_cone(i,3); 
end
figure,imshow(cone_img,[]);
clrmap = [1 0 0;0 1 0;0 0 1];
rgb_mos= label2rgb(uint8(cone_img),clrmap);
figure, imshow(rgb_mos);
% figure,imshow(cone_img,clrmap);
%%
figure, imshowpair(cone_img(:,:), image1(:,:),'montage');
%result = reshape(spectimg,2,[]);
C = imfuse(cone_img,image1,'checkerboard','Scaling','joint');
imshow(C)
%% Homework-4
rd=struct2table(load('ObsFunctions.mat','CIE2006_LMS_2Deg'));
lms=interp1(rd.CIE2006_LMS_2Deg.lambda, rd.CIE2006_LMS_2Deg.data,w5nm');
% one_cone(:,4)=1:104;
l=repmat(reshape(lms(:,1),1,1,61),1080,1920,1);
m=repmat(reshape(lms(:,2),1,1,61),1080,1920,1);
s=repmat(reshape(lms(:,3),1,1,61),1080,1920,1);
% cone_Lres= zeros(104,1); %response of cones
% cone_Mres= zeros(104,1);
% cone_Sres= zeros(104,1);
cone_res=zeros(104,1);
L_img=zeros(1080,1920);
M_img=zeros(1080,1920);
S_img=zeros(1080,1920);

for i=1:104
    cne=((x - one_cone(i,2)).^2 + (y - one_cone(i,1)).^2 <= r.^2);%(1080*1920)for entire103 cone
    switch one_cone(i,3) % (1,3)-l ;(2,3)-m;(3,3)-s
        case 1
            spec_cur_cne= repmat(cne,1,1,61).*l;% 1080*1920*61 ( each pixel*l values)
            cone_res(i,:)= sum(o5nm.*spec_cur_cne,'all'); % only l value for each pixel(104*1) 
        case 2
            spec_cur_cne= repmat(cne,1,1,61).*m;
            cone_res(i,:)= sum(o5nm.*spec_cur_cne,'all');% only m value
%           imshow(cone_res,[])
        case 3
            spec_cur_cne= repmat(cne,1,1,61).*s;
            cone_res(i,:)= sum(o5nm.*spec_cur_cne,'all'); % only s value
%           imshow(cone_res,[])
    end
end
% cone_Lres = cone_Lres/max(cone_Lres(:,1));
% cone_Mres = cone_Mres/max(cone_Mres(:,1));
cone_res = cone_res/max(cone_res(:,1));
%images
for i=1:104
    cne1=((x - one_cone(i,2)).^2 + (y - one_cone(i,1)).^2 <= r.^2);%(1080*1920)for entire104 cone
    switch one_cone(i,3) % (1,3)-l ;(2,3)-m;(3,3)-s
        case 1
            L_img=L_img+cne1.*cone_res(i,1);
        case 2
            M_img=M_img+cne1.*cone_res(i,1);
        case 3
            S_img=S_img+cne1.*cone_res(i,1);
    end
end

figure, imshow(L_img,[]);
figure, imshow(M_img,[]);
figure, imshow(S_img,[]);
fullcne_img= cat(3,L_img,M_img,S_img);
figure, imshow(fullcne_img)
%% Question 5
% one_cone(:,4)=1:104;
% one_cone(:,5)= 1/6;
% one_cone(:,6)= cone_Lres;
% one_cone(:,7)= cone_Mres;
% one_cone(:,8)= cone_Sres;
rg_on = [];
rg_off = [];
by_on = [];
by_off = [];
lum_on = [];
lum_off = [];
for i = 1:104
cone_now = repmat(one_cone(i,1:2),104,1);% current cone
dis = (one_cone(:,1)-cone_now(:,1)).^2+(one_cone(:,2)-cone_now(:,2)).^2; %distance between current cone
neighbor_now = (dis>0&dis<=((r*2)^2+6));% getting current neighbors in range
if sum(neighbor_now)==6 %to check if they have 6 neighbors
lum_on = cat(1,lum_on,[i,sum(cone_res(neighbor_now,1)*(-1/6))+cone_res(i,1)]);%1-1/6
lum_off = cat(1,lum_off,[i,sum(cone_res(neighbor_now,1)*(1/6))-cone_res(i,1)]);%1+1/6
switch one_cone(i,3) %taking labels
    case 1
    rgnsrd = one_cone(neighbor_now,3)==2;
    n = sum(rgnsrd);
    if n
    rg_on = cat(1,rg_on,[i,sum(cone_res(rgnsrd,1)*(-1/n))+cone_res(i,1)]);%L cone center(L-1/no.ofM)
    rg_off = cat(1,rg_off,[i,sum(cone_res(rgnsrd,1)*(1/n))-cone_res(i,1)]);%L cone center(L+1/no.ofM)
    end
    case 3
    bynsrd = (one_cone(neighbor_now,3)==1 |one_cone(neighbor_now,3)==2 );
    n = sum(bynsrd);
    if n
    by_on = cat(1,by_on,[i,sum(cone_res(bynsrd,1)*(-1/n))+cone_res(i,1)]);%S cone center(L-1/no.ofL,M)
    by_off = cat(1,by_off,[i,sum(cone_res(bynsrd,1)*(1/n))-cone_res(i,1)]);%S cone center(L+1/no.ofL,M)
    end
end
end
end
%%
max_val = max(cat(1,lum_on(:,2),lum_off(:,2),rg_on(:,2),rg_off(:,2),by_on(:,2),by_off(:,2)));
min_val = min(cat(1,lum_on(:,2),lum_off(:,2),rg_on(:,2),rg_off(:,2),by_on(:,2),by_off(:,2)));
lumn_im = zeros(1080,1920);
lumf_im = zeros(1080,1920);
rgn_im = zeros(1080,1920);
rgf_im = zeros(1080,1920);
byn_im = zeros(1080,1920);
byf_im = zeros(1080,1920);
for i = 1:104
lumi = ismember(i,lum_on(:,1));
rgi = ismember(i,rg_on(:,1));
byi = ismember(i,by_on(:,1));
current_cone = ((x - one_cone(i,2)).^2 + (y -one_cone(i,1)).^2 < r.^2);
if lumi
    i1 = find(lum_on(:,1)==i);
    lumn_im = lumn_im+current_cone*((lum_on(i1,2)-min_val)/(max_val-min_val));
    lumf_im = lumf_im+current_cone*((lum_off(i1,2)-min_val)/(max_val-min_val));
end
if rgi
    i2 = find(rg_on(:,1)==i);
    rgn_im = rgn_im+current_cone*((rg_on(i2,2)-min_val)/(max_val-min_val));
    rgf_im = rgf_im+current_cone*((rg_off(i2,2)-min_val)/(max_val-min_val));
end
if byi
    i3 = find(by_on(:,1)==i);
    byn_im = byn_im+current_cone*((by_on(i3,2)-min_val)/(max_val-min_val));
    byf_im = byf_im+current_cone*((by_off(i3,2)-min_val)/(max_val-min_val));
end
end
%%
figure,imshow(lumn_im,[]);
figure,imshow(lumf_im,[]);
figure,imshow(cat(3,rgn_im,zeros(1080,1920),zeros(1080,1920)));
figure,imshow(cat(3,zeros(1080,1920),rgf_im,zeros(1080,1920)));
figure,imshow(cat(3,zeros(1080,1920),zeros(1080,1920),byn_im));
figure,imshow(cat(3,byf_im*0.5,byf_im*0.5,zeros(1080,1920)));

%% q5 draft
% Repmatting to 1080,1920
% one_cone(:,4)=1:104;
% one_cone(:,5)= 1/6;
% one_cone(:,6)= cone_Lres;
% one_cone(:,7)= cone_Mres;
% one_cone(:,8)= cone_Sres;
% for i= 1:104
%     aa(i)= (((one_cone(1,1)- one_cone(i,1))^2 + (one_cone(1,2)- one_cone(i,2))^2)<(72*2)^2);  
% end
%  
% n_loc= zeros(104,1);
% for j=1:104
% 
%     if sum(aa(j,:)) < 6
%     else
%         n_loc(j,:)=aa(j);
%     end
% end
% 
% %+center- [sum(surround)/6]=0
% lumon_img= zeros(1080,1920); %1-1/6
% lumoff_img= zeros(1080,1920); %1+1/6
% rg_img= zeros(1080,1920);%L cone center(L-1/no.ofM)
% gr_img= zeros(1080,1920);%L cone center(L+1/no.ofM)
% yb_img= zeros(1080,1920);%S cone center(L-1/no.ofL,M)
% by_img= zeros(1080,1920);%S cone center(L-1/no.ofL,M)
% % function suround 
% for i= 1:104
%     aa(i)= (((one_cone(1,1)- one_cone(i,1))^2 + (one_cone(1,2)- one_cone(i,2))^2)<(72*2)^2);
% end
%%

function x = dimension(width, height, dLambda)

blue = 400;
red = 700;

totWL = round((red-blue)/dLambda) + 1;% tot wavelength 1nm:301,5nm:61

x = zeros(height, width,totWL); %note: this is with pixel_width(1920) columns(1080*1920*301or61)
x1 = zeros(width,totWL);%(1920*301or61)
wL = blue:dLambda:red; %wavelength correspond to dimension 3
wL_C = round(linspace(blue, red, width)); %center wavelengths correspond to rows 
%splitting wavelength using linspace

for j = 1:width
    x1(j,:) = triangle(wL, wL_C(j)); %call triangle function
end %for j=1:1920

x = repmat(x1, 1, 1, height); %clone it in the slice direction 1080 times
x = shiftdim(x,2); %permute the dimensions to make it right

end %function creating spectrum using triangle bandpass



function tri = triangle(wavelength,cenwL)
tri = zeros(size(wavelength)); %initialize the output of the function

Lidx = find((wavelength>=cenwL-5) & (wavelength<=cenwL)); %find  left index of the triangle
Ridx = find((wavelength>=cenwL) & (wavelength<=cenwL+5)); %find right index of the triangle

LwL = wavelength(Lidx); %taking wavelength 
RwL = wavelength(Ridx);

if length(LwL)<6 %Leftmost triangular half line
    tri(Lidx) = (LwL-cenwL)/5 + 1;
    tri(Ridx) =  (cenwL+5-RwL)/5;
elseif length(RwL)<6 %rightmost triangular half line
    tri(Lidx) = (LwL-(cenwL-5))/5;
    tri(Ridx) =  -(RwL-cenwL)/5 + 1;
else %triangles in between
    tri(Lidx) = (LwL-(cenwL-5))/5;
    tri(Ridx) =  ((cenwL+5)-RwL)/5;
end 

end %function to write triangule pulse on the entire spectrum

