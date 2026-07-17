import {
  LayoutDashboard,
  Database,
  Boxes,
  Layers,
  SlidersHorizontal,
  FlaskConical,
  Library,
  History,
  type LucideIcon,
} from "lucide-react";

export interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
  group: string;
}

export const NAV: NavItem[] = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard, group: "Overview" },
  { to: "/datasets", label: "Dataset Manager", icon: Database, group: "Overview" },
  { to: "/pretrain", label: "Pretraining", icon: Boxes, group: "Studios" },
  { to: "/domain", label: "Domain Adaptation", icon: Layers, group: "Studios" },
  { to: "/finetune", label: "Fine-Tuning", icon: SlidersHorizontal, group: "Studios" },
  { to: "/eval", label: "Testing & Eval Lab", icon: FlaskConical, group: "Analysis" },
  { to: "/registry", label: "Model Registry", icon: Library, group: "Analysis" },
  { to: "/runs", label: "Run History", icon: History, group: "Analysis" },
];
