xDoc = xmlread('/path/to/datasets/SVT/svt1/test.xml');

allImageitems = xDoc.getElementsByTagName('image');
n=allImageitems.getLength; bbs=cell(n,1);

for k = 0:allImageitems.getLength-1
   thisImageitem = allImageitems.item(k);

   thisImage = thisImageitem.getElementsByTagName('imageName');
   thisElement = thisImage.item(0);
   fprintf('%s\n',char(thisElement.getFirstChild.getData));
   imname = char(thisElement.getFirstChild.getData);
   fprintf('%s\n',imname(5:9));
   bbs{k+1} = textread(['../data/' imname(5:9)]);

   bbs{k+1} = sortrows(bbs{k+1},5);
   tmp = bbs{k+1}(:,1:4);              
   [tmp2,ia,ic] = unique(tmp,'rows');
   bbs{k+1} = bbs{k+1}(ia,:);
   bbs{k+1} = sortrows(bbs{k+1},5);
end

save('boxes/TextProposals-SVT-FULL-test','bbs')
