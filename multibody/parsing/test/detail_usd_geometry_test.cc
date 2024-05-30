#include "drake/multibody/parsing/detail_usd_geometry.h"

#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usdGeom/capsule.h"
#include "pxr/usd/usdGeom/cube.h"
#include "pxr/usd/usdGeom/cylinder.h"
#include "pxr/usd/usdGeom/mesh.h"
#include "pxr/usd/usdGeom/sphere.h"
#include "pxr/usd/usdGeom/xform.h"
#include <gtest/gtest.h>

#include "drake/common/is_approx_equal_abstol.h"
#include "drake/common/test_utilities/diagnostic_policy_test_base.h"
#include "drake/multibody/parsing/detail_usd_parser.h"

namespace drake {
namespace multibody {
namespace internal {
namespace {

class UsdGeometryTest : public test::DiagnosticPolicyTestBase {
 public:
  UsdGeometryTest() {
    // Construct a parser object so that it initializes the USD library.
    UsdParserWrapper parser;

    stage_ = pxr::UsdStage::CreateInMemory();
    meters_per_unit_ = 0.01;
    stage_up_axis_ = pxr::TfToken("Z");
  }
 protected:
  pxr::UsdStageRefPtr stage_;
  pxr::TfToken stage_up_axis_;
  double meters_per_unit_;
};

TEST_F(UsdGeometryTest, BoxParsingTest) {
  pxr::UsdGeomCube box = pxr::UsdGeomCube::Define(
    stage_, pxr::SdfPath("/Box"));

  double size = 49.0;
  EXPECT_TRUE(box.CreateSizeAttr().Set(size));

  pxr::VtVec3fArray extent;
  EXPECT_TRUE(pxr::UsdGeomCube::ComputeExtent(size, &extent));
  EXPECT_TRUE(box.CreateExtentAttr().Set(extent));

  pxr::GfVec3d scale_factor = pxr::GfVec3d(0.4, 0.5, 0.6);
  auto scale_op = box.AddScaleOp(pxr::UsdGeomXformOp::PrecisionDouble);
  EXPECT_TRUE(scale_op.Set(scale_factor));

  std::optional<Eigen::Vector3d> dimension = GetBoxDimension(
    box.GetPrim(), meters_per_unit_, diagnostic_policy_);
  EXPECT_TRUE(dimension.has_value());

  Eigen::Vector3d correct_dimension =
    UsdVec3dToEigen(scale_factor) * size * meters_per_unit_;
  EXPECT_EQ(dimension.value(), correct_dimension);

  auto shape = CreateGeometryBox(box.GetPrim(), meters_per_unit_,
    diagnostic_policy_);
  EXPECT_TRUE(shape != nullptr);
  geometry::Box* drake_box = dynamic_cast<geometry::Box*>(shape.get());
  EXPECT_EQ(drake_box->size(), correct_dimension);
}

TEST_F(UsdGeometryTest, EllipsoidParsingTest) {
  pxr::UsdGeomSphere ellipsoid = pxr::UsdGeomSphere::Define(
    stage_, pxr::SdfPath("/Ellipsoid"));

  double radius = 17.0;
  EXPECT_TRUE(ellipsoid.CreateRadiusAttr().Set(radius));

  pxr::VtVec3fArray extent;
  EXPECT_TRUE(pxr::UsdGeomSphere::ComputeExtent(radius, &extent));
  EXPECT_TRUE(ellipsoid.CreateExtentAttr().Set(extent));

  pxr::GfVec3d scale_factor = pxr::GfVec3d(0.6, 1.1, 2.9);
  auto scale_op = ellipsoid.AddScaleOp(pxr::UsdGeomXformOp::PrecisionDouble);
  EXPECT_TRUE(scale_op.Set(scale_factor));

  std::optional<Eigen::Vector3d> dimension = GetEllipsoidDimension(
    ellipsoid.GetPrim(), meters_per_unit_, diagnostic_policy_);
  EXPECT_TRUE(dimension.has_value());

  Eigen::Vector3d correct_dimension =
    UsdVec3dToEigen(scale_factor) * radius * meters_per_unit_;
  EXPECT_EQ(dimension.value(), correct_dimension);

  auto shape = CreateGeometryEllipsoid(ellipsoid.GetPrim(), meters_per_unit_,
    diagnostic_policy_);
  EXPECT_TRUE(shape != nullptr);
  geometry::Ellipsoid* drake_ellipsoid =
    dynamic_cast<geometry::Ellipsoid*>(shape.get());
  auto actual_dimension = Eigen::Vector3d(
    drake_ellipsoid->a(), drake_ellipsoid->b(), drake_ellipsoid->c());
  EXPECT_EQ(actual_dimension, correct_dimension);
}

TEST_F(UsdGeometryTest, CylinderParsingTest) {
  pxr::UsdGeomCylinder cylinder = pxr::UsdGeomCylinder::Define(
    stage_, pxr::SdfPath("/Cylinder"));

  double radius = 62.0;
  double height = 199.0;
  pxr::TfToken axis = pxr::TfToken("Z");
  EXPECT_TRUE(cylinder.CreateRadiusAttr().Set(radius));
  EXPECT_TRUE(cylinder.CreateHeightAttr().Set(height));
  EXPECT_TRUE(cylinder.CreateAxisAttr().Set(axis));

  pxr::VtVec3fArray extent;
  EXPECT_TRUE(pxr::UsdGeomCylinder::ComputeExtent(
    height, radius, axis, &extent));
  EXPECT_TRUE(cylinder.CreateExtentAttr().Set(extent));

  pxr::GfVec3d scale_factor = pxr::GfVec3d(0.7, 0.7, 0.9);
  auto scale_op = cylinder.AddScaleOp(pxr::UsdGeomXformOp::PrecisionDouble);
  EXPECT_TRUE(scale_op.Set(scale_factor));

  std::optional<Eigen::Vector2d> dimension = GetCylinderDimension(
    cylinder.GetPrim(), meters_per_unit_, stage_up_axis_, diagnostic_policy_);
  EXPECT_TRUE(dimension.has_value());

  double correct_radius = scale_factor[0] * radius * meters_per_unit_;
  double correct_height = scale_factor[2] * height * meters_per_unit_;
  auto correct_dimension = Eigen::Vector2d(correct_radius, correct_height);
  EXPECT_EQ(dimension.value(), correct_dimension);

  auto shape = CreateGeometryCylinder(cylinder.GetPrim(), meters_per_unit_,
    stage_up_axis_, diagnostic_policy_);
  EXPECT_TRUE(shape != nullptr);
  geometry::Cylinder* drake_cylinder =
    dynamic_cast<geometry::Cylinder*>(shape.get());
  auto actual_dimension = Eigen::Vector2d(
    drake_cylinder->radius(), drake_cylinder->length());
  EXPECT_EQ(actual_dimension, correct_dimension);
}

TEST_F(UsdGeometryTest, CapsuleParsingTest) {
  pxr::UsdGeomCapsule capsule = pxr::UsdGeomCapsule::Define(
    stage_, pxr::SdfPath("/Capsule"));

  double radius = 101;
  double height = 45;
  pxr::TfToken axis = pxr::TfToken("Z");
  EXPECT_TRUE(capsule.CreateRadiusAttr().Set(radius));
  EXPECT_TRUE(capsule.CreateHeightAttr().Set(height));
  EXPECT_TRUE(capsule.CreateAxisAttr().Set(axis));

  pxr::VtVec3fArray extent;
  EXPECT_TRUE(pxr::UsdGeomCapsule::ComputeExtent(
    height, radius, axis, &extent));
  EXPECT_TRUE(capsule.CreateExtentAttr().Set(extent));

  pxr::GfVec3d scale_factor = pxr::GfVec3d(0.7, 0.7, 0.9);
  auto scale_op = capsule.AddScaleOp(pxr::UsdGeomXformOp::PrecisionDouble);
  EXPECT_TRUE(scale_op.Set(scale_factor));

  std::optional<Eigen::Vector2d> dimension = GetCapsuleDimension(
    capsule.GetPrim(), meters_per_unit_, stage_up_axis_, diagnostic_policy_);
  EXPECT_TRUE(dimension.has_value());

  double correct_radius = scale_factor[0] * radius * meters_per_unit_;
  double correct_height = scale_factor[2] * height * meters_per_unit_;
  auto correct_dimension = Eigen::Vector2d(correct_radius, correct_height);
  EXPECT_EQ(dimension.value(), correct_dimension);

  auto shape = CreateGeometryCapsule(capsule.GetPrim(), meters_per_unit_,
    stage_up_axis_, diagnostic_policy_);
  EXPECT_TRUE(shape != nullptr);
  geometry::Capsule* drake_capsule =
    dynamic_cast<geometry::Capsule*>(shape.get());
  auto actual_dimension = Eigen::Vector2d(
    drake_capsule->radius(), drake_capsule->length());
  EXPECT_EQ(actual_dimension, correct_dimension);
}

TEST_F(UsdGeometryTest, MeshParsingTest) {
  pxr::UsdGeomMesh mesh = pxr::UsdGeomMesh::Define(
    stage_, pxr::SdfPath("/Mesh"));

  // The following specifies an octahedron mesh.
  auto vertices = pxr::VtArray<pxr::GfVec3f>{
    pxr::GfVec3f(1, 0, 0), pxr::GfVec3f(0, -1, 0),
    pxr::GfVec3f(-1, 0, 0), pxr::GfVec3f(0, 1, 0),
    pxr::GfVec3f(0, 0, 1), pxr::GfVec3f(0, 0, -1)};
  auto face_vertex_counts = pxr::VtArray<int>{3, 3, 3, 3, 3, 3, 3, 3};
  auto face_vertex_indices = pxr::VtArray<int>{
    1, 0, 4, 2, 1, 4, 3, 2, 4, 0, 3, 4, 0, 1, 5, 1, 2, 5, 2, 3, 5, 3, 0, 5};
  double scale_factor = 129.2;

  EXPECT_TRUE(mesh.CreatePointsAttr().Set(vertices));
  EXPECT_TRUE(mesh.CreateFaceVertexCountsAttr().Set(face_vertex_counts));
  EXPECT_TRUE(mesh.CreateFaceVertexIndicesAttr().Set(face_vertex_indices));

  auto scale_op = mesh.AddScaleOp(pxr::UsdGeomXformOp::PrecisionDouble);
  EXPECT_TRUE(scale_op.Set(
    pxr::GfVec3d(scale_factor, scale_factor, scale_factor)));

  auto shape = CreateGeometryMesh("octahedron.obj", mesh.GetPrim(),
    meters_per_unit_, diagnostic_policy_);
  EXPECT_TRUE(shape != nullptr);
  geometry::Mesh* drake_mesh = dynamic_cast<geometry::Mesh*>(shape.get());
  EXPECT_EQ(drake_mesh->scale(), scale_factor);

  // Check whether Drake can sucessfully parse that file by computing the
  // convex hull of the octahedron mesh.
  auto convex_hull = drake_mesh->GetConvexHull();
  EXPECT_EQ(convex_hull.num_faces(), 8);
}

TEST_F(UsdGeometryTest, GetRigidTransformTest) {
  pxr::UsdGeomXform xform = pxr::UsdGeomXform::Define(
    stage_, pxr::SdfPath("/Xform"));

  pxr::GfVec3d translation = pxr::GfVec3d(196, 51, 133.1);
  pxr::GfVec3d rotation_xyz = pxr::GfVec3d(21.59, -9.56, 155);
  auto translate_op = xform.AddTranslateOp(
    pxr::UsdGeomXformOp::PrecisionDouble);
  auto rotate_op = xform.AddRotateXYZOp(
    pxr::UsdGeomXformOp::PrecisionDouble);
  EXPECT_TRUE(translate_op.Set(translation));
  EXPECT_TRUE(rotate_op.Set(rotation_xyz));

  auto transform = GetPrimRigidTransform(
    xform.GetPrim(), meters_per_unit_, diagnostic_policy_);
  EXPECT_TRUE(transform.has_value());
  auto actual_translation = transform.value().translation();
  auto actual_rotation_xyz =
    transform.value().rotation().ToRollPitchYaw().vector();

  auto intended_translation = UsdVec3dToEigen(translation * meters_per_unit_);
  auto intended_rotation_xyz = UsdVec3dToEigen(rotation_xyz * (M_PI / 180.0));
  EXPECT_TRUE(is_approx_equal_abstol(
    actual_translation, intended_translation, 1e-10));
  EXPECT_TRUE(is_approx_equal_abstol(
    actual_rotation_xyz, intended_rotation_xyz, 1e-10));
}

}  // namespace
}  // namespace internal
}  // namespace multibody
}  // namespace drake
