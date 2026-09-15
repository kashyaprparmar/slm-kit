import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FolderKanban, Plus } from "lucide-react";
import { api } from "@/lib/api";
import { useWorkflow } from "@/lib/workflow";
import { PageHeader } from "@/components/PageHeader";
import { ErrorPanel } from "@/components/ErrorPanel";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";

export default function Projects() {
  const queryClient = useQueryClient();
  const projects = useQuery({ queryKey: ["projects"], queryFn: api.listProjects });
  const [activeProject, setActiveProject] = useWorkflow<number | null>("active-project", null);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const create = useMutation({
    mutationFn: api.createProject,
    onSuccess: (project) => {
      setActiveProject(project.id);
      setName("");
      setDescription("");
      queryClient.invalidateQueries({ queryKey: ["projects"] });
    },
  });
  const remove = useMutation({
    mutationFn: api.deleteProject,
    onSuccess: (_, id) => {
      if (activeProject === id) setActiveProject(null);
      queryClient.invalidateQueries({ queryKey: ["projects"] });
    },
  });
  useEffect(() => {
    if (projects.data && activeProject != null && !projects.data.some((project) => project.id === activeProject)) {
      setActiveProject(null);
    }
  }, [projects.data, activeProject, setActiveProject]);

  return <div className="space-y-6">
    <PageHeader title="Projects" description="Persistent workspaces keep model, dataset, configuration, runs, and evaluation context together." />
    {(projects.error || create.error || remove.error) && <ErrorPanel error={projects.error || create.error || remove.error} retry={() => projects.refetch()} />}
    <div className="grid gap-5 lg:grid-cols-[360px_1fr]">
      <Card><CardHeader><CardTitle>Create project</CardTitle></CardHeader><CardContent className="space-y-3">
        <Input aria-label="Project name" placeholder="Project name" value={name} onChange={(event) => setName(event.target.value)} />
        <Textarea aria-label="Project description" placeholder="Goal, dataset, or experiment notes" value={description} onChange={(event) => setDescription(event.target.value)} />
        <Button className="w-full" disabled={!name.trim() || create.isPending} onClick={() => create.mutate({ name: name.trim(), description: description.trim() })}>
          <Plus className="mr-2 size-4" />{create.isPending ? "Creating…" : "Create and select"}
        </Button>
      </CardContent></Card>
      <div className="grid content-start gap-3 sm:grid-cols-2">
        {projects.data?.map((project) => <Card key={project.id} className={activeProject === project.id ? "border-primary" : ""}>
          <CardContent className="space-y-3 p-4">
            <div className="flex items-start justify-between gap-3"><div className="flex items-center gap-2"><FolderKanban className="size-4 text-primary" /><span className="font-semibold">{project.name}</span></div>{activeProject === project.id && <Badge variant="success">Active</Badge>}</div>
            <p className="min-h-10 text-sm text-muted-foreground">{project.description || "No description."}</p>
            <p className="text-xs text-muted-foreground">{project.run_count ?? 0} linked runs · Updated {new Date(project.updated_at).toLocaleString()}</p>
            <div className="flex gap-2">
              <Button size="sm" variant={activeProject === project.id ? "secondary" : "default"} disabled={activeProject === project.id} onClick={() => setActiveProject(project.id)}>{activeProject === project.id ? "Selected" : "Open project"}</Button>
              <Button size="sm" variant="ghost" disabled={remove.isPending} onClick={() => {
                if (window.confirm(`Delete project “${project.name}”? Runs and artifacts will be retained.`)) remove.mutate(project.id);
              }}>Delete workspace</Button>
            </div>
          </CardContent>
        </Card>)}
        {!projects.isLoading && !projects.data?.length && <Card><CardContent className="p-6 text-sm text-muted-foreground">Create a project to preserve context across the training lifecycle. Existing runs remain usable without one.</CardContent></Card>}
      </div>
    </div>
  </div>;
}
