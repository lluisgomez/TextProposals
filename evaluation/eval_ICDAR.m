%% run and evaluate on entire dataset (see boxesData.m and boxesEval.m)
split='test'; data=boxesData_ICDAR2013('split',split);

%calculate IoU recall rates and plot results
nm=cell({'TextProposals-ICDAR-FULL'});
boxesEval('data',data,'names',nm,'thrs',.5,'show',2,'fName','ICDAR2013_2');
boxesEval('data',data,'names',nm,'thrs',.7,'show',3,'fName','ICDAR2013_3');
boxesEval('data',data,'names',nm,'thrs',.9,'show',4,'fName','ICDAR2013_4');
boxesEval('data',data,'names',nm,'thrs',.5:.05:1,'cnts',1000,'show',5,'fName','ICDAR2013_5');
boxesEval('data',data,'names',nm,'thrs',.5:.05:1,'cnts',5000,'show',6,'fName','ICDAR2013_6');
boxesEval('data',data,'names',nm,'thrs',.5:.05:1,'cnts',10000,'show',7,'fName','ICDAR2013_7');
