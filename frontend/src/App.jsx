import "./App.css";
import GraphTrackerV2 from "./Components/DashboardV2/GraphTrackerV2";
import {
  BrowserRouter as Router,
  Routes,
  Route,
} from "react-router-dom";
import { Toaster } from 'react-hot-toast';

function App() {
  return (
    <>
      <Toaster />
      <Router>
        <Routes>
          <Route path="/" element={<GraphTrackerV2 />} />
          {/* Fallback to render the system on any other path */}
          <Route path="*" element={<GraphTrackerV2 />} />
        </Routes>
      </Router>
    </>
  );
}

export default App;
