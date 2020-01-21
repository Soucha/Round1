No installation

The Learner can solve only Micro-tasks from 5.3 to 9 (and should be able to solve 14 and 15).
	It may crash on the others.

The Learner initialized by remote_receiver.py and listens on '127.0.0.1', port=5556

The Learner tries to load brainPrelearned.pkl by default
 - this can by changed by editing remove_receiver.py on line 57 and providing argument:
	"" - the Learner starts without prelearned model, or
	filename of a pickled brain (EFSM hierarchy).
 - the hierarchy is stored in brainTmp.pkl after an instance is solved

The second optional argument for the Learner is a name of file to which the Learner stores the hierarchy in DOT language.
	The file then can be visualized in FSMvis.html (large models take time to show).