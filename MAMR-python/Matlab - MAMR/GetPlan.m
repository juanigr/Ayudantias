function pi = GetPlan(p,q,R,S)
pi = (1/S)*repmat(p,1,S) + (1/R)*repmat(q',R,1) - 1/(S*R);
return