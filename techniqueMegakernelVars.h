namespace Megakernel
{
	struct globalvarsT
	{
		volatile int doneCounter;
		volatile int endCounter;

		volatile int maxConcurrentBlocks; //=0
		volatile int maxConcurrentBlockEvalDone;//=0
	};
}

