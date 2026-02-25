window.mermaidConfig = {
  startOnLoad: false,
  theme: 'default',
};

document$.subscribe(() => {
  mermaid.initialize(window.mermaidConfig);
  mermaid.run({ querySelector: '.mermaid' });
});
