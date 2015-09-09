%% run and evaluate on entire dataset (see boxesData.m and boxesEval.m)
split='test'; data=boxesData_SVT('split',split);

nm=cell({'TextProposals-SVT-FULL'});
boxesEval('data',data,'names',nm,'thrs',.5,'show',2,'fName','SVT_2');
boxesEval('data',data,'names',nm,'thrs',.7,'show',3,'fName','SVT_3');
boxesEval('data',data,'names',nm,'thrs',.9,'show',4,'fName','SVT_4');
boxesEval('data',data,'names',nm,'thrs',.5:.05:1,'cnts',1000,'show',5,'fName','SVT_5');
boxesEval('data',data,'names',nm,'thrs',.5:.05:1,'cnts',5000,'show',6,'fName','SVT_6');
boxesEval('data',data,'names',nm,'thrs',.5:.05:1,'cnts',10000,'show',7,'fName','SVT_7');
