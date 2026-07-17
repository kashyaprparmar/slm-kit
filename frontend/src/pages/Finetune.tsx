import { TrainingStudio } from "@/components/studio/TrainingStudio";

export default function Finetune() {
  return (
    <TrainingStudio
      task="finetune"
      title="Fine-Tuning Studio"
      description="LoRA / QLoRA / DoRA / full fine-tuning with live 8GB fit checks."
      datasetKinds={["instruction"]}
      defaultOutputName="my-finetune"
    />
  );
}
