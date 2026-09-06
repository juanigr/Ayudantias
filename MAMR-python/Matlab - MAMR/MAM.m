function [p,val,cpu,theta] = MAM(d,q,M,R,S,p,rho,UseGPU,tol,MaxCPU,PrintEvery)
%--------------------------------------------------------------------------
%             Initialization
%--------------------------------------------------------------------------
t0   = tic;
proj_simplex = @(y) max(y-max((cumsum(sort(y,1,'descend'),1)-1)./(1:size(y,1))'),0);
Kn   = round(sqrt(R));
% Initializing theta 
theta= cell(M,1);
pk   = cell(R,M);
% Move data to GPU
if UseGPU
    avg  = gpuArray.zeros(R,1);
    p    = gpuArray(p);
    for m = 1:M
        q{m} = gpuArray(q{m});
        pk{m}= gpuArray(pk{m});
        d{m} = gpuArray(d{m});
        theta{m} = gpuArray(theta{m});
    end
else
    disp('No gpu!!!!!!!!!!!!!')
    avg=zeros(R,1);
end
a    = (1./S)';
a    = a/sum(a);
for m=1:M
   theta{m}= GetPlan(p,q{m}',R,S(m));
   pk{m}   = p;
 %  theta{m}= -d{m}/rho;pk{m}=sum(theta{m},2);avg=avg + pk{m}*a(m);
end
%p   = avg - (sum(avg)-1)/R;
avg  = p;
val  = 0;  
%--------------------------------------------------------------------------
%          Main Loop
%--------------------------------------------------------------------------
NextPrint = 0; cpu = 0;k=0;
while cpu <= MaxCPU
     k = k+1;
     cpu = toc(t0);
     if cpu >= NextPrint 
         nx = norm(p-avg);
         fprintf(1,'k = %5.0f, |pk-pkk| = %5.2e, cpu = %5.0f \n',k,nx,toc(t0)); 
         imagesc(reshape(1-p,Kn,Kn));
         title(strcat('k=',num2str(k),', t=',num2str(NextPrint)));
         name = strcat('Fig\',num2str(NextPrint),'-MAM.png');
         colormap hot;
         salvaPNG(gcf,name)
         pause(0.01) % Pause cafe
        if nx<=tol
            break
        end
        NextPrint = NextPrint+PrintEvery;
    end
    p   = avg;         
    avg = zeros(R,1);
    for m=1:M   
        pihat    = proj_simplex( (theta{m} + 2*(p-pk{m})/S(m) - (1/rho)*d{m})./q{m} ).*q{m} ;
        theta{m} = pihat - (p-pk{m})/S(m);
        pk{m}    = sum(theta{m},2);
        avg      = avg + a(m)*pk{m}; 
    end         
end
cpu = toc(t0);
fprintf(1,'k = %5.0f,|pk-pkk| = %5.2e, cpu = %5.0f \n',k,nx,cpu); 

imagesc(reshape(1-p,Kn,Kn));
title(strcat('k=',num2str(k),', t=',num2str(round(cpu))));
name = strcat('Fig\',num2str(round(cpu)),'-MAM.png');
colormap hot;
salvaPNG(gcf,name)
pause(0.01) % Pause cafe
return
