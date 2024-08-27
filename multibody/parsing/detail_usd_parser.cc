#include "drake/multibody/parsing/detail_usd_parser.h"

#include <filesystem>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "pxr/base/plug/registry.h"
#include "pxr/base/tf/token.h"
#include "pxr/usd/usd/primRange.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usdGeom/capsule.h"
#include "pxr/usd/usdGeom/cube.h"
#include "pxr/usd/usdGeom/cylinder.h"
#include "pxr/usd/usdGeom/mesh.h"
#include "pxr/usd/usdGeom/sphere.h"
#include "pxr/usd/usdGeom/xform.h"
#include "pxr/usd/usdPhysics/distanceJoint.h"
#include "pxr/usd/usdPhysics/fixedJoint.h"
#include "pxr/usd/usdPhysics/joint.h"
#include "pxr/usd/usdPhysics/prismaticJoint.h"
#include "pxr/usd/usdPhysics/revoluteJoint.h"
#include "pxr/usd/usdPhysics/sphericalJoint.h"
#include <fmt/format.h>

#include "drake/common/find_runfiles.h"
#include "drake/common/temp_directory.h"
#include "drake/common/unused.h"
#include "drake/multibody/parsing/detail_common.h"
#include "drake/multibody/parsing/detail_make_model_name.h"
#include "drake/multibody/parsing/detail_usd_geometry.h"

namespace drake {
namespace multibody {
namespace internal {

namespace fs = std::filesystem;

struct UsdStageMetadata {
    double meters_per_unit = 1.0;
    pxr::TfToken up_axis = pxr::TfToken("Z");
};

class UsdParser {
 public:
  explicit UsdParser(const ParsingWorkspace& ws);
  ~UsdParser() = default;
  std::vector<ModelInstanceIndex> AddAllModels(
    const DataSource& data_source,
    const std::optional<std::string>& parent_model_name);

 private:
  void ProcessFreeFloatingRigidBody(const pxr::UsdPrim& prim, bool is_static);
  SpatialInertia<double> GetSpatialInertiaOfPrim(const pxr::UsdPrim& prim);
  void ProcessArticulation(const pxr::UsdPrim& prim);
  void ProcessLink(const pxr::UsdPrim& prim,
    ModelInstanceIndex model_instance);
  void ProcessJoint(const pxr::UsdPhysicsJoint& joint);
  std::set<pxr::SdfPath> FindPhysicsArticulationRoots();
  UsdStageMetadata GetStageMetadata();
  std::unique_ptr<geometry::Shape> CreateCollisionGeometry(
    const pxr::UsdPrim& prim);
  std::unique_ptr<geometry::Shape> CreateVisualGeometry(
    const pxr::UsdPrim& prim);
  const RigidBody<double>* CreateRigidBody(const pxr::UsdPrim& prim,
    ModelInstanceIndex model_instance);
  void RaiseUnsupportedPrimTypeError(const pxr::UsdPrim& prim);
  std::pair<std::optional<pxr::UsdPrim>, std::optional<pxr::UsdPrim>>
    GetBody0Body1ForJoint(const pxr::UsdPhysicsJoint& joint);

  std::string GetRigidBodyName(const pxr::UsdPrim& prim) {
    return fmt::format("{}-RigidBody", prim.GetPath().GetString());
  }
  std::string GetVisualGeometryName(const pxr::UsdPrim& prim) {
    return fmt::format("{}-VisualGeometry", prim.GetPath().GetString());
  }
  std::string GetCollisionGeometryName(const pxr::UsdPrim& prim) {
    return fmt::format("{}-CollisionGeometry", prim.GetPath().GetString());
  }
  std::string GetArticulationName(const pxr::UsdPrim& prim) {
    return fmt::format("{}-Articulation", prim.GetPath().GetString());
  }
  std::string GetJointName(const pxr::UsdPrim& prim) {
    return fmt::format("{}-Joint", prim.GetPath().GetString());
  }

  std::string temp_directory_;
  inline static std::vector<std::string> mesh_files_;
  const ParsingWorkspace& w_;
  pxr::UsdStageRefPtr stage_;
  UsdStageMetadata metadata_;
  std::vector<ModelInstanceIndex> model_instances_;
  ModelInstanceIndex free_bodies_instance_;
};

UsdParserWrapper::UsdParserWrapper() = default;

UsdParserWrapper::~UsdParserWrapper() = default;

void UsdParserWrapper::InitializeOpenUsdLibrary() {
  // Register all relevant plugins.
  auto& registry = pxr::PlugRegistry::PlugRegistry::GetInstance();
  std::vector<std::string> json_paths{{
      "openusd_internal/pxr/usd/ar/plugInfo.json",
      "openusd_internal/pxr/usd/ndr/plugInfo.json",
      "openusd_internal/pxr/usd/sdf/plugInfo.json",
      "openusd_internal/pxr/usd/usd/plugInfo.json",
      "openusd_internal/pxr/usd/usdGeom/plugInfo.json",
      "openusd_internal/pxr/usd/usdPhysics/plugInfo.json",
      "openusd_internal/pxr/usd/usdShade/plugInfo.json",
  }};
  for (const auto& json_path : json_paths) {
    const RlocationOrError location = FindRunfile(json_path);
    if (!location.error.empty()) {
      throw std::runtime_error(location.error);
    }
    const fs::path info_dir = fs::path(location.abspath).parent_path();
    registry.RegisterPlugins(info_dir.string());
  }
}

std::optional<ModelInstanceIndex> UsdParserWrapper::AddModel(
    const DataSource& data_source, const std::string& model_name,
    const std::optional<std::string>& parent_model_name,
    const ParsingWorkspace& workspace) {
  unused(data_source, model_name, parent_model_name, workspace);
  throw std::runtime_error("UsdParser::AddModel is not implemented.");
}

std::vector<ModelInstanceIndex> UsdParserWrapper::AddAllModels(
    const DataSource& data_source,
    const std::optional<std::string>& parent_model_name,
    const ParsingWorkspace& workspace) {
  UsdParser parser(workspace);
  return parser.AddAllModels(data_source, parent_model_name);
}

UsdParser::UsdParser(const ParsingWorkspace& ws) : w_{ws} {
  temp_directory_ = temp_directory();

  // The first time AddAllModels is called, we need to call
  // InitializeOpenUsdLibrary() to prepare.  We can ensure that happens
  // exactly once, in a threadsafe manner, by using a dummy function-local
  // static variable.
  static const int ignored = []() {
    UsdParserWrapper::InitializeOpenUsdLibrary();
    return 0;
  }();
  unused(ignored);
}

std::unique_ptr<geometry::Shape> UsdParser::CreateVisualGeometry(
  const pxr::UsdPrim& prim) {
  if (prim.IsA<pxr::UsdGeomCube>()) {
    return CreateGeometryBox(
      prim, metadata_.meters_per_unit, w_.diagnostic);
  } else if (prim.IsA<pxr::UsdGeomSphere>()) {
    return CreateGeometryEllipsoid(
      prim, metadata_.meters_per_unit, w_.diagnostic);
  } else if (prim.IsA<pxr::UsdGeomCapsule>()) {
    return CreateGeometryCapsule(
      prim, metadata_.meters_per_unit, metadata_.up_axis, w_.diagnostic);
  } else if (prim.IsA<pxr::UsdGeomCylinder>()) {
    return CreateGeometryCylinder(
      prim, metadata_.meters_per_unit, metadata_.up_axis, w_.diagnostic);
  } else if (prim.IsA<pxr::UsdGeomMesh>()) {
    // TODO(#15263): Here we create an obj file for each mesh and pass the
    // filename into the constructor of geometry::Mesh. It is a temporary
    // solution while #15263 is being worked on. This is something we must fix
    // before we enable this parser in the default build options.
    std::string obj_file_path = fmt::format("{}/{}.obj", temp_directory_,
      mesh_files_.size());
    mesh_files_.push_back(obj_file_path);
    return CreateGeometryMesh(
      obj_file_path, prim, metadata_.meters_per_unit, w_.diagnostic);
  } else {
    RaiseUnsupportedPrimTypeError(prim);
    return nullptr;
  }
}

std::unique_ptr<geometry::Shape> UsdParser::CreateCollisionGeometry(
  const pxr::UsdPrim& prim) {
  // For now, we use the raw visual geometry for collision detection
  // for all geometry types.
  return CreateVisualGeometry(prim);
}

SpatialInertia<double> UsdParser::GetSpatialInertiaOfPrim(
  const pxr::UsdPrim& prim) {
  // TODO(hong-nvidia): Check if the Prim defines its inertia explicitly.
  // if (prim.HasAPI(pxr::TfToken("PhysicsMassAPI"))) {
  //   pxr::UsdAttribute inertia_attribute

  //   pxr::UsdAttribute inertia_attribute =
  //     pxr::UsdPhysicsMassAPI(prim).GetDiagonalInertiaAttr();
  //   pxr::GfVec3f diagonal_inertia;
  //   if (inertia_attribute.Get(&diagonal_inertia)) {

  //     // return static_cast<double>(mass);
  //   }
  // }

  std::optional<SpatialInertia<double>> inertia;
  if (prim.IsA<pxr::UsdGeomCube>()) {
    inertia = CreateSpatialInertiaForBox(
      prim, metadata_.meters_per_unit, w_.diagnostic);
  } else if (prim.IsA<pxr::UsdGeomSphere>()) {
    inertia = CreateSpatialInertiaForEllipsoid(
      prim, metadata_.meters_per_unit, w_.diagnostic);
  } else if (prim.IsA<pxr::UsdGeomCapsule>()) {
    inertia = CreateSpatialInertiaForCapsule(
      prim, metadata_.meters_per_unit, metadata_.up_axis, w_.diagnostic);
  } else if (prim.IsA<pxr::UsdGeomCylinder>()) {
    inertia = CreateSpatialInertiaForCylinder(
      prim, metadata_.meters_per_unit, metadata_.up_axis, w_.diagnostic);
  }  // else {
  //   RaiseUnsupportedPrimTypeError(prim);
  // }

  // TODO(hong-nvidia): The following lines are temporary. To be fixed soon.
  if (inertia.has_value()) {
    return inertia.value();
  } else {
    // w_.diagnostic.Warning(fmt::format("Failed to parse SpatialInertia for "
    // "the Prim at {}.", prim.GetPath().GetString()));
    return SpatialInertia<double>::Zero();
  }
}

const RigidBody<double>* UsdParser::CreateRigidBody(const pxr::UsdPrim& prim,
  ModelInstanceIndex model_instance) {
  SpatialInertia<double> inertia = GetSpatialInertiaOfPrim(prim);
  return &w_.plant->AddRigidBody(
    GetRigidBodyName(prim), model_instance, inertia);
}

void UsdParser::ProcessFreeFloatingRigidBody(const pxr::UsdPrim& prim,
  bool is_static) {
  auto collision_geometry = CreateCollisionGeometry(prim);
  if (!collision_geometry) {
    w_.diagnostic.Error(fmt::format("Failed to create collision "
      "geometry for prim at {}.", prim.GetPath().GetString()));
    return;
  }

  auto visual_geometry = CreateVisualGeometry(prim);
  if (!visual_geometry) {
    w_.diagnostic.Error(fmt::format("Failed to create visual "
      "geometry for prim at {}.", prim.GetPath().GetString()));
    return;
  }

  std::optional<math::RigidTransform<double>> prim_transform =
    GetPrimRigidTransform(prim, metadata_.meters_per_unit, w_.diagnostic);
  if (!prim_transform.has_value()) {
    return;
  }

  const RigidBody<double>* rigid_body;
  // Pose of the geometry in the body frame.
  math::RigidTransform<double> X_BG;
  if (is_static) {
    rigid_body = &w_.plant->world_body();
    X_BG = prim_transform.value();
  } else {
    rigid_body = CreateRigidBody(prim, free_bodies_instance_);
    X_BG = math::RigidTransform<double>::Identity();
    w_.plant->SetDefaultFreeBodyPose(*rigid_body, prim_transform.value());
  }

  if (!rigid_body) {
    w_.diagnostic.Error(fmt::format("Failed to create RigidBody "
      "for prim at {}.", prim.GetPath().GetString()));
    return;
  }
  w_.plant->RegisterCollisionGeometry(
    *rigid_body,
    X_BG,
    *collision_geometry,
    GetCollisionGeometryName(prim),
    GetPrimFriction(prim));

  std::optional<Eigen::Vector4d> prim_color = GetGeomPrimColor(prim,
    w_.diagnostic);

  w_.plant->RegisterVisualGeometry(
    *rigid_body,
    X_BG,
    *visual_geometry,
    GetVisualGeometryName(prim),
    prim_color.has_value() ? prim_color.value() : default_geom_prim_color());
}

void UsdParser::ProcessLink(const pxr::UsdPrim& prim,
  ModelInstanceIndex model_instance) {
  drake::log()->info(fmt::format("  Processing link: {}",
    prim.GetPath().GetString()));

  // TODO(hong-nvidia): Create RigidBody.
  const RigidBody<double>* rigid_body = CreateRigidBody(prim, model_instance);

  // TODO(hong-nvidia): Register Visual Geometries.

  // TODO(hong-nvidia): Register Collision Geometries.
}

std::pair<std::optional<pxr::UsdPrim>, std::optional<pxr::UsdPrim>>
  UsdParser::GetBody0Body1ForJoint(
  const pxr::UsdPhysicsJoint& joint) {
  const std::string prim_path = joint.GetPrim().GetPath().GetString();

  std::optional<pxr::UsdPrim> body0 = std::nullopt;
  std::optional<pxr::UsdPrim> body1 = std::nullopt;

  std::vector<pxr::SdfPath> body0_targets;
  bool success = joint.GetBody0Rel().GetTargets(&body0_targets);
  if (success && body0_targets.size() == 1) {
    body0 = stage_->GetPrimAtPath(body0_targets[0]);
  }

  std::vector<pxr::SdfPath> body1_targets;
  success = joint.GetBody1Rel().GetTargets(&body1_targets);
  if (success && body1_targets.size() == 1) {
    body1 = stage_->GetPrimAtPath(body1_targets[0]);
  }

  return std::make_pair(body0, body1);
}

void UsdParser::ProcessJoint(const pxr::UsdPhysicsJoint& joint) {
  pxr::UsdPrim prim = joint.GetPrim();
  if (prim.HasAPI(pxr::TfToken("PhysicsArticulationRootAPI"))) {
    // TODO(hong-nvidia): Determine what to do with root joint.
    return;
  }

  const std::string prim_path = joint.GetPrim().GetPath().GetString();
  drake::log()->info(fmt::format("  Processing joint: {}", prim_path));

  auto body0_body1_pair = GetBody0Body1ForJoint(joint);
  if (!body0_body1_pair.first.has_value() ||
      !body0_body1_pair.second.has_value()) {
    drake::log()->error(fmt::format("Joint at {} has invalid reference to ",
      "either body 0 or body1.", prim_path));
    return;
  }

  drake::log()->info(fmt::format("    Body0: {}",
    body0_body1_pair.first.value().GetPath()));
  drake::log()->info(fmt::format("    Body1: {}",
    body0_body1_pair.second.value().GetPath()));

  std::string parent_body_name = GetRigidBodyName(
    body0_body1_pair.first.value());
  std::string child_body_name = GetRigidBodyName(
    body0_body1_pair.second.value());

  // auto joint = WeldJoint<double>(
  //   GetJointName(joint.GetPrim()),
  // );

  if (prim.IsA<pxr::UsdPhysicsFixedJoint>()) {
    w_.plant->AddJoint<WeldJoint>(
      GetJointName(joint.GetPrim()),
      w_.plant->GetRigidBodyByName(parent_body_name),
      std::nullopt,  // TODO(hong-nvidia): Implement this
      w_.plant->GetRigidBodyByName(child_body_name),
      std::nullopt,  // TODO(hong-nvidia): Implement this
      math::RigidTransformd::Identity());
  } else if (prim.IsA<pxr::UsdPhysicsPrismaticJoint>()) {
    // TODO(hong-nvidia): Implement this.
  } else if (prim.IsA<pxr::UsdPhysicsRevoluteJoint>()) {
    // TODO(hong-nvidia): Implement this.
  } else if (prim.IsA<pxr::UsdPhysicsSphericalJoint>()) {
    // TODO(hong-nvidia): Implement this.
  } else if (prim.IsA<pxr::UsdPhysicsDistanceJoint>()) {
    // TODO(hong-nvidia): Implement this.
  } else {
    // TODO(hong-nvidia): Implement this.
    // Plain joint with manual specification of constraints?
  }
}

void UsdParser::ProcessArticulation(const pxr::UsdPrim& prim) {
  drake::log()->info(fmt::format("Processing articulation: {}",
    prim.GetPath().GetString()));

  ModelInstanceIndex articulation_instance = w_.plant->AddModelInstance(
    GetArticulationName(prim));
  model_instances_.push_back(articulation_instance);

  std::vector<pxr::UsdPrim> links;
  std::vector<pxr::UsdPhysicsJoint> joints;

  for (const pxr::UsdPrim& component : prim.GetChildren()) {
    if (component.IsA<pxr::UsdGeomXform>() &&
        component.HasAPI(pxr::TfToken("PhysicsRigidBodyAPI"))) {
      links.push_back(component);
      // A link can contain a joint as its child, so we check if that is the
      // case here.
      for (const pxr::UsdPrim& subcomponent : component.GetChildren()) {
        if (subcomponent.IsA<pxr::UsdPhysicsJoint>()) {
          joints.push_back(pxr::UsdPhysicsJoint(subcomponent));
        }
      }
    } else if (component.IsA<pxr::UsdPhysicsJoint>()) {
      joints.push_back(pxr::UsdPhysicsJoint(component));
    } else {
      // Prim is neither a link or a joint, so we ignore it as it is redundant
      // in the subtree of this articulation.
      continue;
    }
  }

  for (const pxr::UsdPrim& link : links) {
    ProcessLink(link, articulation_instance);
  }

  for (const pxr::UsdPhysicsJoint& joint : joints) {
    ProcessJoint(joint);
  }
}

UsdStageMetadata UsdParser::GetStageMetadata() {
  UsdStageMetadata metadata;

  bool success = false;
  if (stage_->HasAuthoredMetadata(pxr::TfToken("metersPerUnit"))) {
    success = stage_->GetMetadata(pxr::TfToken("metersPerUnit"),
      &metadata.meters_per_unit);
  }
  if (!success) {
    w_.diagnostic.Warning(fmt::format(
      "Failed to read metersPerUnit in stage metadata. "
      "Using the default value '{}' instead.", metadata.meters_per_unit));
  }

  success = false;
  if (stage_->HasAuthoredMetadata(pxr::TfToken("upAxis"))) {
    success = stage_->GetMetadata(pxr::TfToken("upAxis"), &metadata.up_axis);
  }
  if (!success) {
    w_.diagnostic.Warning(fmt::format(
      "Failed to read upAxis in stage metadata. "
      "Using the default value '{}' instead.", metadata.up_axis));
  }
  if (metadata.up_axis != "Z") {
    throw std::runtime_error("Parsing for Y-up or X-up stage is not yet "
      "implemented.");
  }
  return metadata;
}

std::set<pxr::SdfPath> UsdParser::FindPhysicsArticulationRoots() {
  std::set<pxr::SdfPath> articulation_root_paths;
  for (const pxr::UsdPrim& prim : stage_->Traverse()) {
    if (prim.HasAPI(pxr::TfToken("PhysicsArticulationRootAPI"))) {
      // If the API is applied on a joint, then the root of the articulation is
      // the parent Prim of what this joint is pointing to.
      if (prim.IsA<pxr::UsdPhysicsJoint>()) {
        auto body0_body1_pair = GetBody0Body1ForJoint(
          pxr::UsdPhysicsJoint(prim));
        if (!body0_body1_pair.second.has_value()) {
          w_.diagnostic.Error(fmt::format("Failed to read the `body1` "
            "attribute of the joint at {}.", prim.GetPath().GetString()));
          continue;
        }
        pxr::SdfPath target_link_path =
          body0_body1_pair.second.value().GetPath();
        pxr::SdfPath articulation_root_path = target_link_path.GetParentPath();
        articulation_root_paths.insert(articulation_root_path);
      } else {  // Otherwise, the current Prim is the root of the articulation.
        articulation_root_paths.insert(prim.GetPath());
      }
    }
  }
  return articulation_root_paths;
}

std::vector<ModelInstanceIndex> UsdParser::AddAllModels(
  const DataSource& data_source,
  const std::optional<std::string>& parent_model_name) {
  if (data_source.IsFilename()) {
    std::string file_absolute_path = data_source.GetAbsolutePath();
    if (!std::filesystem::exists(file_absolute_path)) {
      w_.diagnostic.Error(
        fmt::format("File does not exist: {}.", file_absolute_path));
      return std::vector<ModelInstanceIndex>();
    }
    stage_ = pxr::UsdStage::Open(file_absolute_path);
    if (!stage_) {
      w_.diagnostic.Error(fmt::format("Failed to open USD stage: {}.",
        data_source.filename()));
      return std::vector<ModelInstanceIndex>();
    }
  } else {
    stage_ = pxr::UsdStage::CreateInMemory();
    if (!stage_->GetRootLayer()->ImportFromString(data_source.contents())) {
      w_.diagnostic.Error(fmt::format("Failed to load in-memory USD stage."));
      return std::vector<ModelInstanceIndex>();
    }
  }

  metadata_ = GetStageMetadata();

  std::string model_name = MakeModelName(
    data_source.GetStem(), parent_model_name, w_);
  free_bodies_instance_ = w_.plant->AddModelInstance(model_name);
  model_instances_.push_back(free_bodies_instance_);

  std::set<pxr::SdfPath> articulation_root_paths =
    UsdParser::FindPhysicsArticulationRoots();

  // BFS traversal of the scene graph and process Prims.
  std::queue<pxr::UsdPrim> prim_queue;
  prim_queue.push(stage_->GetPseudoRoot());
  while (!prim_queue.empty()) {
    pxr::UsdPrim current_prim = prim_queue.front();
    prim_queue.pop();

    std::set<pxr::SdfPath>::iterator element_position =
      articulation_root_paths.find(current_prim.GetPath());
    if (element_position != articulation_root_paths.end()) {
      // Current Prim is the root of an articulation.
      ProcessArticulation(current_prim);
      articulation_root_paths.erase(element_position);
    } else {
      // Current Prim is outside of any articulation subtree. Check if it is
      // a free-floating rigid body.
      if (current_prim.HasAPI(pxr::TfToken("PhysicsCollisionAPI"))) {
        drake::log()->info(fmt::format("Processing environment object {}",
          current_prim.GetPath().GetString()));
        if (current_prim.HasAPI(pxr::TfToken("PhysicsRigidBodyAPI"))) {
          // If the Prim has the collision API but not the RigidBodyAPI, then
          // it is considered a regular free-floating rigid body.
          ProcessFreeFloatingRigidBody(current_prim, false);
        } else {
          // If the Prim has the collision API but not the RigidBodyAPI, then
          // it is considered a static collider.
          ProcessFreeFloatingRigidBody(current_prim, true);
        }
      }

      // Explore the remainder of the subtree.
      for (const auto& child : current_prim.GetChildren()) {
        prim_queue.push(child);
      }
    }
  }

  // TODO(hong-nvidia): Remove DRAKE_DEMAND()
  DRAKE_DEMAND(articulation_root_paths.size() == 0);

  return model_instances_;
}

void UsdParser::RaiseUnsupportedPrimTypeError(const pxr::UsdPrim& prim) {
  pxr::TfToken prim_type = prim.GetTypeName();
  if (prim_type == "") {
    w_.diagnostic.Error(fmt::format("The type of the Prim at {} is "
      "not specified. Please assign a type to it.",
      prim.GetPath().GetString()));
  } else {
    w_.diagnostic.Error(fmt::format("Unsupported Prim type '{}' at {}.",
      prim_type, prim.GetPath().GetString()));
  }
}

}  // namespace internal
}  // namespace multibody
}  // namespace drake
