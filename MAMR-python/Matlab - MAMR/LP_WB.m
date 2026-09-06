function [p,val,cpu,m,n]=LP_WB(D,Q,M,R,S)
t0=tic;
c=[];
A=[];
b=[];
W=[];
for m=1:M
    c   = [c;reshape(D{m},R*S(m),1)];
    aux = [kron(speye(S(m)),ones(1,R))
          kron(ones(1,S(m)),speye(R))  ];
     auxW = [zeros(S(m),R);-eye(R)];
     W  = [W;auxW];
     A  = blkdiag(A,aux);
     b  = [b;[Q{m}';zeros(R,1)]];
end
c = [c;zeros(R,1)];
A = [A,W];
[m,n]=size(A);
opts=optimset('Display','iter');
[sol,val] = linprogGurobi(c,[],[],A,b,zeros(n,1),inf(n,1),[],opts);
p = sol(n-R+1:end);
cpu=toc(t0);
return