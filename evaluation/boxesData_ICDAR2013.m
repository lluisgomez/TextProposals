function data = boxesData_ICDAR2013( varargin )
% Get ground truth data for object proposal bounding box evaluation.
%

% get default parameters (unimportant parameters are undocumented)
dfs={ 'resDir','boxes/', 'dataDir','boxes/ICDAR2013/', 'split','test' };
o=getPrmDflt(varargin,dfs,1);

% check if data already exists, if yes load and return
dataNm=[o.resDir '/GroundTruth' '-' 'ICDAR2013' '-' o.split '.mat'];
if(exist(dataNm,'file')), data=load(dataNm); data=data.data; return; end

% locations of ICDAR2013 dataset
if(~exist(o.dataDir,'dir')), error('dataset not found, see help'); end
gtDir=o.dataDir;

% generate list of ids, image and gt filenames then load gt

if(~exist(fullfile(gtDir,[o.split '_locations.xml']),'file')), error('dataset not found, see help'); end
xDoc = xmlread(fullfile(gtDir,[o.split '_locations.xml']));

allImageitems = xDoc.getElementsByTagName('image');
n=allImageitems.getLength; ids=cell(n,1); imgs=cell(n,1); gt=cell(n,1);

for k = 0:allImageitems.getLength-1
   thisImageitem = allImageitems.item(k);

   thisImage = thisImageitem.getElementsByTagName('imageName');
   thisElement = thisImage.item(0);
   %fprintf('%s\n',char(thisElement.getFirstChild.getData));
   ids{k+1} = char(thisElement.getFirstChild.getData);
   imgs{k+1} = [gtDir '/test/' char(thisElement.getFirstChild.getData)];

   allRectangleitems = thisImageitem.getElementsByTagName('taggedRectangle');
   bbs = zeros(allRectangleitems.getLength,5);
   for kk = 0:allRectangleitems.getLength-1
     thisRectangleitem = allRectangleitems.item(kk);
     x = thisRectangleitem.getAttribute('x');
     y = thisRectangleitem.getAttribute('y');
     w = thisRectangleitem.getAttribute('width');
     h = thisRectangleitem.getAttribute('height');
     bbs(kk+1,:) = [str2num(char(x)) str2num(char(y)) str2num(char(w)) str2num(char(h)) 0];
     %fprintf(' %d %d %d %d\n',str2num(char(x)),str2num(char(y)),str2num(char(w)),str2num(char(h)));
   end
   gt{k+1} = bbs;
end


% create output structure and cache to disk
data=struct('split',o.split,'n',n,'ids',{ids},'imgs',{imgs},'gt',{gt});
if(~exist(o.resDir,'dir')), mkdir(resDir); end; save(dataNm,'data');

end
