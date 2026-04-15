import argparse
from multiprocessing import Manager, Process

from scripts.LoadDataset import DatasetLoader
from scripts.Worker import APIWorker, QueueWorker, LearningWorker, dataset_feeder

def parse_args():
	"""Parse command-line arguments for the runner.

	Args:
		None

	Returns:
		Parsed argparse namespace.
	"""
	parser = argparse.ArgumentParser(description="RiskSceneGraph runner with API")
	parser.add_argument("--host", type=str, default="0.0.0.0", help="API host")
	parser.add_argument("--port", type=int, default=8005, help="API port")
	parser.add_argument("--usedataset", action="store_true", help="Use dataset feeder instead of only API queue")
	parser.add_argument("--path", type=str, default=None, help="Dataset base path (overrides config)")
	parser.add_argument("--vlm-offline", action="store_true", help="Use DummyVLMHelper instead of real VLM")
	parser.add_argument("--no-sam-rap", dest="use_sam_rap", action="store_false", default=True, help="Disable SAM segmentation and Visual RAP")
	parser.add_argument("--save-output", action="store_true", default=False, help="Save output files (graphs, visualizations)")
	parser.add_argument("--small-vlm", action="store_true", default=False, help="Enable small-VLM mode (strip gt/distance and filter duplicates)")
	return parser.parse_args()


def main():
	"""Start API server, processing workers, and optional dataset feeder.

	Args:
		None

	Returns:
		None
	"""
	args = parse_args()
	print("args:", args)
	mgr = Manager()
	work_queue = mgr.Queue(maxsize=128)
	learning_queue = mgr.Queue(maxsize=256)
	shared_state = mgr.dict()
	shared_state["latest"] = {"nodes": [], "edges": []}

	# Start API server process
	api = APIWorker(queue=work_queue, shared_state=shared_state, host=args.host, port=args.port)
	api_proc = Process(target=api.run, name="api-server", daemon=True)
	api_proc.start()
	print(f"API server is starting on http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}")
	print(f"Docs can be accessed at http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/docs")

	queue_worker = QueueWorker(
		queue=work_queue,
		shared_state=shared_state,
		use_offline=args.vlm_offline,
		use_sam_rap=args.use_sam_rap,
		save_output=args.save_output,
		learning_queue=learning_queue,
		small_vlm=args.small_vlm,
	)
	consumer_proc = Process(target=queue_worker.run, name="consumer", daemon=True)
	consumer_proc.start()

	# Start learning worker if SAM/RAP is enabled
	learning_proc = None
	if args.use_sam_rap:
		learning_worker = LearningWorker(learning_queue=learning_queue, use_offline=args.vlm_offline)
		learning_proc = Process(target=learning_worker.run, name="learning-worker", daemon=True)
		learning_proc.start()

	dataset_proc = None
	if args.usedataset:
		# Load dataset and feed into the queue in a separate process
		d_loader = DatasetLoader(dataset_dir_override=args.path)
		dataset = d_loader.load_dataset()
		print("dataset loaded; feeding to queue")
		dataset_proc = Process(
			target=dataset_feeder,
			kwargs={
				"queue": work_queue,
				"dataset": dataset,
				"shared_state": shared_state,
			},
			name="dataset-feeder",
			daemon=True,
		)
		dataset_proc.start()

	# Handle shutdown
	try:
		if dataset_proc is not None:
			dataset_proc.join()
		consumer_proc.join()
		api_proc.join()
	except KeyboardInterrupt:
		print("Shutting down...")
		try:
			work_queue.put(None)
			if learning_proc is not None:
				learning_queue.put(None)
		except Exception:
			pass
		for p in [dataset_proc, consumer_proc, learning_proc, api_proc]:
			if p is not None and p.is_alive():
				p.terminate()
		for p in [dataset_proc, consumer_proc, learning_proc, api_proc]:
			if p is not None:
				p.join(timeout=5)


if __name__ == "__main__":
	main()









