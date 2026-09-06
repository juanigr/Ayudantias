function salvaPNG(h,filename)
if nargin<2
    filename='Fig.png';
    if nargin==0
        h = gcf;
    end
end
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
print(h,char(filename),'-dpng')
return
