function [p,val,cpu,theta] = MAMR(d,q,M,R,S,p,rho,UseGPU,tol,MaxCPU,PrintEvery)
%--------------------------------------------------------------------------
%             Initialization
%--------------------------------------------------------------------------
t0   = tic;
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
   d{m}    = d{m}/rho; % scaling the distance to avoid uncessary computations
end
%p   = avg - (sum(avg)-1)/R;
avg  = p;
val  = 0; 
%--------------------------------------------------------------------------
%          Main Loop
%--------------------------------------------------------------------------
NextPrint = 0; cpu = 0;k=0;
while cpu<=MaxCPU
     k = k+1;
     cpu = toc(t0);
     if cpu >= NextPrint  
         nx = norm(p-(avg-(sum(avg)-1)/R));
         fprintf(1,'k = %5.0f, |pk-pkk| = %5.2e, cpu = %5.0f \n',k,nx,toc(t0)); 
         imagesc(reshape(1-p,Kn,Kn));
        title(strcat('k=',num2str(k),', t=',num2str(NextPrint)));
         colormap hot;   
         name = strcat('Fig\',num2str(NextPrint),'-MAMR.png'); 
         colormap hot;
         salvaPNG(gcf,name)
         pause(0.01) % Pause cafe
        if nx<=tol
            break
        end
        NextPrint = NextPrint+PrintEvery;
     end
    p   = avg - (sum(avg)-1)/R;
    val = 0;
    avg = zeros(R,1);
    for m=1:M 
        qm      = sum(theta{m},1);
        beta    = (p-pk{m})/S(m)+(q{m}-qm)/R + (sum(qm)-1)/(R*S(m));
        theta{m}= max(theta{m}+beta-d{m},-beta);
        pk{m}   = sum(theta{m},2);
        avg     = avg + a(m)*pk{m};        
    end
end
cpu=toc(t0);
fprintf(1,'k = %5.0f, |pk-pkk| = %5.2e, cpu = %5.0f \n',k,nx,cpu); 

imagesc(reshape(1-p,Kn,Kn));
title(strcat('k=',num2str(k),', t=',num2str(round(cpu))));
colormap hot;    
name = strcat('Fig\',num2str(round(cpu)),'-MAMR.png'); 
colormap hot;
salvaPNG(gcf,name)
pause(0.01) 
return
