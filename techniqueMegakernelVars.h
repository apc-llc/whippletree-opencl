namespace Megakernel
{
	struct globalvarsT
	{
		volatile uint doneCounter;
		volatile uint endCounter;

		volatile uint maxConcurrentBlocks; //=0
		volatile uint maxConcurrentBlockEvalDone;//=0
	};
}

