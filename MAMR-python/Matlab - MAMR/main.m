function main
close all; clc; rng(0)
% Method  = 1 IBP
% Method  = 2 MAM
% Method  = 3 MAM-R   Romero's formula
% Method  = 4 MAM-H   Heuristic
% Method  = 5 LP      Gurobi
Method = [ 3 ];
% DataSet = 1  Peyre
% DataSet = 2  Altschuler

DataSet = 1;
ExactWB = false;
UseGPU  = false;

MaxCPU     = 1*30;  % seconds
PrintEvery = 5; % seconds
tol        = -inf;%1e-6;

M = 10;  % number of images
K = 60; %  images K x K
for met=Method
    %--------------------------------------------------------------------------
    %          Compute the distance
    %--------------------------------------------------------------------------
    fprintf(1,'Computing distance...\n');
    
    if ExactWB
        Kn= M*(K-1) + 1;
        R = Kn*Kn;
        D = (distGrid(K,M).^2)/(K*K);
    else
        Kn = K;
        R  = K*K;
        D  = (distGrid(K,1).^2)/(K*K);
    end
    
    figure
    Q = zeros(K*K,1); 
    
    %--------------------------------------------------------------------------
    %                  Read the data and plot the images
    %--------------------------------------------------------------------------
    fprintf(1,'Reading data...\n');
    
    for i=1:M
        if DataSet ==1
          d = csvread(strcat(pwd,'/dataPeyre/',num2str(i),'.csv')); d(:,3) = d(:,3)/sum(d(:,3));
        else
          d = load(strcat(pwd,'/dataAltschuler/',num2str(i),'.txt')); d(:,[1 2]) = round(K*d(:,[1 2]));
        end
        im= zeros(K,K);
        n = size(d,1);
        for j = 1:n
            ii=d(j,1);
            jj=d(j,2);
            jj=max(jj,1);
            im(ii,jj) = d(j,3);
        end
        Q(:,i) = reshape(im,K*K,1);
        if i<=25
        subplot(5,5,i)
        imagesc(1-im)
        colormap hot
        box on;
        set(gca,'xtick',[],'ytick',[])
        xlabel('');
        ylabel('');
        end
    end
    clear im d
    %%%%%%
    figure
    if met==1
        disp('Running IBP!');
        lambda = 300;%520;%300;
        tic;
        p=bregmanWassersteinBarycenter(Q,D/median(D(:)),MaxCPU,lambda,UseGPU,tol);      
    else
        %--------------------------------------------------------------------------
        %        Arrange the data for MAM
        %--------------------------------------------------------------------------
        fprintf(1,'Arranging data...\n');
        S = [];
        q = cell(M,1);
        d = cell(M,1);
       % p = Q(:,1);
   %     if ExactWB
            p=ones(R,1)/R;
    %    end
        for m=1:M
            I    = Q(:,m)>1e-15;
            S    = [S, sum(I)];
            d{m} = D(:,I);
            q{m} = Q(I,m)';
            q{m} = q{m}/sum(q{m});
        end 
        clear D Q aux 
       
        rho = 5e3%350;
        if met==2
            disp('Running MAM!');
            p = MAM(d,q,M,R,S,p,rho,UseGPU,tol,MaxCPU,PrintEvery);
        elseif met==3
            disp('Running MAM-R!');
            p = MAMR(d,q,M,R,S,p,rho,UseGPU,tol,MaxCPU,PrintEvery);
        elseif met==4
            disp('Running MAM-H!');
            p = MAMH(d,q,M,R,S,p,rho,UseGPU,tol,MaxCPU,PrintEvery);
             %p = MAMHtest(d,q,M,R,S,p,rho,UseGPU,tol,MaxCPU,PrintEvery);
        else
            disp('Running Gurobi!');
            p = LP_WB(d,q,M,R,S);
        end
    end
    p=reshape(p,Kn,Kn);
    imagesc(1-p);
    colormap hot; 
    name = strcat(strcat('Fig\Final-met-',num2str(met),'.png')); 
    salvaPNG(gcf,name)
    close all
end
            
return


