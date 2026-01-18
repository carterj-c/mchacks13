import CSVPlot from "./CSVPlot.jsx";
import ProjectHeader from "./ProjectHeader.jsx";
import XGBModel from "./XGBModel.jsx";
import "./App.css";

function App() {
  return (
    <div className="App">
      <ProjectHeader />
      <XGBModel />
      <CSVPlot />
    </div>
  );
}

export default App;
