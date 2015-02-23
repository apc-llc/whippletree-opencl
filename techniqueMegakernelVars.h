namespace Megakernel
{
	struct globalvarsT
	{
		volatile int doneCounter;
		volatile int endCounter;

		int maxConcurrentBlocks; //=0
		volatile int maxConcurrentBlockEvalDone;//=0
	};
}

